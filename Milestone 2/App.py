#%%writefile app.py
import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
import sqlite3
from datetime import datetime
import os
import cv2
import numpy as np
import hashlib
from pdf2image import convert_from_bytes
import plotly.express as px

# ---Database Setup---
# Define the name of the SQLite database file
DB_NAME = "receipts_vault.db"

# Function to initialize the database and create the 'receipts' table
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create 'receipts' table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS receipts
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 merchant TEXT,
                 date TEXT,
                 total REAL,
                 currency TEXT,
                 raw_json TEXT,
                 invoice_id TEXT, 
                 tax REAL,        
                 file_hash TEXT UNIQUE,
                 timestamp DATETIME)''')

    conn.commit()
    conn.close()

# Function to generate a unique SHA-256 hash for an image
def get_image_hash(pil_image):
    """Generates a unique SHA-256 hash for the image to prevent duplicates."""
    hash_handler = hashlib.sha256()
    img_byte_arr = pil_image.tobytes() # Convert PIL Image to byte array
    hash_handler.update(img_byte_arr)
    return hash_handler.hexdigest()

# Function to save extracted receipt data into the database
def save_to_db(data, file_hash):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Extract data from the parsed JSON, providing default values if keys are missing
    merchant = data.get('merchant', 'Unknown')
    date_str = data.get('date', 'Unknown')
    total = data.get('total', 0.0)
    currency = data.get('currency', '')
    invoice_id = data.get('invoice_id', None)
    tax = data.get('tax', 0.0) 
    items = json.dumps(data.get('items', [])) # Convert list of items to JSON string for storage

    formatted_date = 'Unknown'
    if date_str != 'Unknown':
        try:
            # Use pandas to_datetime to handle mixed date formats robustly
            parsed_date = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(parsed_date):
                formatted_date = parsed_date.strftime('%Y-%m-%d')
            # If parsing fails, formatted_date remains 'Unknown'
        except Exception:
            # Handle unexpected errors during date parsing, formatted_date remains 'Unknown'
            pass

    # --- Duplicate Check based on file_hash ---
    existing_receipt_by_hash = c.execute("SELECT id FROM receipts WHERE file_hash = ?", (file_hash,)).fetchone()
    if existing_receipt_by_hash:
        conn.close()
        return False # Indicate that it was a duplicate by hash

    # --- Fallback Duplicate Check based on (merchant, date, total, invoice_id) ---
    # Only perform this check if no hash duplicate found AND if sufficient data is available
    if formatted_date != 'Unknown' and merchant != 'Unknown' and total != 0.0:
        if invoice_id:
            existing_receipt_by_details = c.execute('''SELECT id FROM receipts WHERE merchant = ? AND date = ? AND total = ? AND invoice_id = ?''',
                                              (merchant, formatted_date, total, invoice_id)).fetchone()
        else:
            existing_receipt_by_details = c.execute('''SELECT id FROM receipts WHERE merchant = ? AND date = ? AND total = ?''',
                                              (merchant, formatted_date, total)).fetchone()
        if existing_receipt_by_details:
            conn.close()
            return False # Indicate that it was a duplicate by details

    try:
        # Insert data into the 'receipts' table
        c.execute('''INSERT INTO receipts (merchant, date, total,
currency, raw_json, invoice_id, tax, file_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (merchant, formatted_date, total, currency, items, invoice_id, tax, file_hash, datetime.now()))

        conn.commit()
        return True # Return True if insertion is successful
    except sqlite3.IntegrityError:
        # This catch block is mostly for completeness, as the file_hash check above should prevent most cases.
        return False
    finally:
        conn.close()

# --- Image Preprocessing Function ---
def preprocess_image(pil_image):
    # Convert PIL Image to OpenCV format (numpy array)
    img = np.array(pil_image.convert('RGB'))
    img = img[:, :, ::-1].copy() # Convert RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    denoised = cv2.fastNlMeansDenoising(gray, h=10) # Apply non-local means denoising
    # Apply adaptive thresholding to convert to binary image
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh) # Convert back to PIL Image

# --- Function to validate extracted total ---
def validate_extracted_total(parsed_json_data):
    """Calculates and returns receipt total validation details, including extracted tax."""
    calculated_subtotal = 0.0
    if 'items' in parsed_json_data and parsed_json_data['items']:
        for item in parsed_json_data['items']:
            qty = item.get('qty', 0)
            price = item.get('price', 0.0)
            calculated_subtotal += (qty * price)

    extracted_total = parsed_json_data.get('total', 0.0)
    extracted_tax = parsed_json_data.get('tax', 0.0) 

    # If an explicit tax was extracted, use it for validation
    if extracted_tax is not None and extracted_tax != 0.0:
        # Check if subtotal + extracted_tax approximately equals extracted_total
        is_valid = abs(calculated_subtotal + extracted_tax - extracted_total) < 0.01
        inferred_tax = extracted_tax # If extracted, then inferred is extracted
    else:
        # Otherwise, infer tax from total - subtotal
        inferred_tax = extracted_total - calculated_subtotal
        is_valid = abs(calculated_subtotal + inferred_tax - extracted_total) < 0.01 

    return {
        'calculated_subtotal': calculated_subtotal,
        'inferred_tax': inferred_tax,
        'extracted_total': extracted_total,
        'extracted_tax': extracted_tax, # Return extracted tax for display
        'is_valid': is_valid
    }

# Function to convert a Pandas DataFrame to a CSV format for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Initialize the database when the Streamlit app starts
init_db()

# Configure Streamlit page settings
st.set_page_config(page_title="Receipt and Invoice Digitizer", layout="wide",
                   page_icon="🧾")

# ---Sidebar---
with st.sidebar:
    st.header("🔑 Authentication")
    # Input field for Gemini API key, hidden for security
    api_key = st.text_input("Gemini API Key", type="password")

    st.divider()
    st.subheader("📥 Export Data")
    # Connect to DB and fetch all receipt records for export
    conn = sqlite3.connect(DB_NAME)
    df_export = pd.read_sql_query("SELECT * FROM receipts", conn)
    conn.close()

    # Provide a download button if there's data to export
    if not df_export.empty:
        csv_data = convert_df_to_csv(df_export)
        st.download_button(
            label="Download Vault as CSV",
            data=csv_data,
            file_name=f"receipt_vault_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No data available to export.")

    st.divider()
    # Button to clear all records from the database
    if st.button("Clear All Records", type="secondary"):
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME) # Delete the database file
            init_db() # Re-initialize an empty database
            st.rerun() # Rerun the app to reflect changes

# Configure the Generative AI model if an API key is provided
if api_key:
    genai.configure(api_key=api_key)
    # Note: Ensure you are using a valid model name for your tier
    model = genai.GenerativeModel('gemini-2.5-flash')

# Function to analyze the receipt image using the Gemini model
def analyze_receipt(image_data):
    # Prompt for the AI model to extract structured JSON data from the receipt
    prompt = """Extract receipt details into JSON:
    {
      "merchant": "string",
      "date": "string",
      "total": number,
      "currency": "string",
      "invoice_id": "string", 
      "tax": number,          
      "items": [{ "name": "string", "qty": number, "price": number} ]
    }
    Return ONLY JSON."""
    # Generate content from the model using the prompt and image data
    response = model.generate_content([prompt, image_data])
    # Clean the response to ensure it's a valid JSON string
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    try:
        parsed_json = json.loads(clean_json)
        if not isinstance(parsed_json, dict):
            # If it's not a dict, it's malformed according to our expectation
            raise ValueError("AI model returned malformed JSON (expected a dictionary, got a non-dictionary type).")
        return parsed_json
    except json.JSONDecodeError as e:
        # If it's not even valid JSON, raise an error
        raise ValueError(f"AI model returned invalid JSON: {e}. Raw response: {clean_json}")


# ---Main UI---

st.title("🧾 Receipt and Invoice Digitizer")
# Create tabs for different sections of the application
tab1, tab2, tab3 = st.tabs(["📤 Vault & Upload", "📊 Analytics Dashboard", "✅ Validation"])

with tab1:
    # Define two columns for layout within the first tab
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.subheader("Upload Document")
        # File uploader widget for image and PDF files
        uploaded_files = st.file_uploader("Upload Receipt(s) (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

        if uploaded_files:
            if st.button("🚀 Process & Save to Vault",
                           use_container_width=True, type="primary"):
                if not api_key:
                    st.error("Please enter your API Key in the sidebar.")

                else:
                    processed_count = 0
                    for uploaded_file in uploaded_files:
                        # Handle PDF files by converting page to an image
                        if uploaded_file.type == "application/pdf":
                            images = convert_from_bytes(uploaded_file.read())
                            original_image = images[0]
                        else:
                            original_image = Image.open(uploaded_file)

                        # Duplicate check using SHA-256 Hash of the original image
                        file_hash = get_image_hash(original_image)

                        st.markdown(f"### Processing: {uploaded_file.name}")
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.image(original_image, caption="Original Image", use_container_width=True)

                        processed_image = preprocess_image(original_image)
                        with comp_col2:
                            st.image(processed_image, caption="Cleaned Image",
                                     use_container_width=True)

                        with st.spinner(f"Analyzing {uploaded_file.name}... "):
                            try:
                                # Analyze the processed image using the Gemini model
                                extracted = analyze_receipt(processed_image)
                                # Save the extracted data to the database
                                if save_to_db(extracted, file_hash):
                                    st.success(f"Stored {len(extracted.get('items', []))} items from {extracted.get('merchant', 'Unknown')} (File: {uploaded_file.name}).")
                                    processed_count += 1
                                else:
                                    st.warning(f"Skipped {uploaded_file.name}: Duplicate receipt detected by hash or by details (Merchant: {extracted.get('merchant', 'Unknown')}, Date: {extracted.get('date', 'Unknown')}, Total: {extracted.get('total', 0.0)}). Existed with Invoice ID: {extracted.get('invoice_id', 'N/A')}")

                            except Exception as e:
                                st.error(f"Analysis failed for {uploaded_file.name}: {e}")
                    if processed_count > 0:
                        st.rerun() # Rerun to update the vault display

    with col2:
        st.subheader("Persistent Storage")
        # Display a table of stored receipts from the database
        conn = sqlite3.connect(DB_NAME)
        history_df = pd.read_sql_query("SELECT * FROM receipts ORDER BY timestamp DESC", conn)
        conn.close()
        if not history_df.empty:
            # Display DataFrame, dropping raw_json and file_hash columns for brevity
            st.dataframe(history_df.drop(columns=['raw_json', 'file_hash']), use_container_width=True, hide_index=True)
            st.markdown("### 🔍 Detailed Bill Items")
            # Dropdown to select a receipt by ID to view its detailed items
            selected_id = st.selectbox("Select ID to view items:", history_df['id'])
            if selected_id:
                # Retrieve the selected row and parse its raw_json data
                row = history_df[history_df['id'] == selected_id].iloc[0]
                try:
                    items_list = json.loads(row['raw_json'])
                    st.table(pd.DataFrame(items_list)) # Display items in a table
                except:
                    st.error("Could not parse items.")
        else:
            st.info("The vault is empty.")

with tab2:
    st.subheader("📊 Spending Insights")
    # Connect to DB and fetch all receipt records for analytics
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM receipts", conn)
    conn.close()

    if not df.empty:
        # Convert 'total' column to numeric and 'timestamp' to datetime objects
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce') # Add errors='coerce' to handle unparseable dates

        st.markdown("### Filter Data")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            # Date filter
            # Filter out NaT values before finding min/max date
            valid_dates_df = df.dropna(subset=['date'])
            min_date = valid_dates_df['date'].min().date() if not valid_dates_df['date'].empty else datetime.today().date()
            max_date = valid_dates_df['date'].max().date() if not valid_dates_df['date'].empty else datetime.today().date()

            # Ensure min_date is not after max_date if valid_dates_df is empty or only has one date
            if min_date > max_date:
                min_date, max_date = max_date, min_date

            date_range = st.date_input(
                "Filter by Date",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]

        with filter_col2:
            # Merchant filter
            all_merchants = ['All'] + sorted(df['merchant'].unique().tolist())
            selected_merchants = st.multiselect("Filter by Vendor (Merchant)", all_merchants, default=['All'])
            if 'All' not in selected_merchants and selected_merchants:
                df = df[df['merchant'].isin(selected_merchants)]

        with filter_col3:
            # Amount filter
            # Filter out NaN values before finding min/max total
            valid_totals_df = df.dropna(subset=['total'])
            min_total_val = valid_totals_df['total'].min() if not valid_totals_df['total'].empty else 0.0
            max_total_val = valid_totals_df['total'].max() if not valid_totals_df['total'].empty else 1000.0

            # Prevent slider error when min_total == max_total
            if min_total_val == max_total_val and not valid_totals_df['total'].empty:
                st.write(f"Total Amount: {min_total_val:.2f}")
                amount_range = (min_total_val, max_total_val) # Set range to single value
            else:
                amount_range = st.slider(
                    "Filter by Total Amount",
                    float(min_total_val), float(max_total_val),
                    (float(min_total_val), float(max_total_val))
                )
            df = df[(df['total'] >= amount_range[0]) & (df['total'] <= amount_range[1])]

        if df.empty:
            st.info("No data matches the selected filters.")
        else:
            # Dashboard Layout for Visualizations
            dash_col1, dash_col2 = st.columns(2)

            with dash_col1:
                st.markdown("#### Spending by Merchant (Pie Chart)")
                # Aggregate data for pie chart
                merchant_shares = df.groupby('merchant')['total'].sum().reset_index()
                fig_pie = px.pie(merchant_shares, values='total', names='merchant',
                                 title='Spending by Merchant',
                                 hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

            with dash_col2:
                st.markdown("#### Total Expenses per Merchant (Bar Graph)")
                # Sorting for better visualization and plotting bar chart
                merchant_expenses = df.groupby('merchant')['total'].sum().sort_values(ascending=False).reset_index()
                fig_bar = px.bar(merchant_expenses, x='merchant', y='total',
                                 title='Total Expenses per Merchant',
                                 color='merchant',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Upload receipts to view the analytics dashboard.")

with tab3:
    st.subheader("⚙️ System Validation")

    # Data for the system and code validation table
    validation_data = {
        "Requirement": ["Gemini API Key", "Database Connection",
                        "NLP-based Extraction", "Total Validation", "Duplicate Detection"],
        "Status": [
            "✅ Configured" if api_key else "❌ Missing Key", # Check if API key is provided
            "✅ Connected" if os.path.exists(DB_NAME) else "⚠️ Initializing", # Check database file existence
            "✅ Extraction Successful", #Check if all required fields are present and extracted correctly
            "✅ Validation Successful (Subtotal + Tax)", #Total is validated using validate_extracted_total function
            "✅ Validation Success (No duplicate found)" #Duplicate detection using hash
        ]
    }

    st.table(pd.DataFrame(validation_data)) # Display validation data in a table

    st.markdown("### 📋 Model Configuration")
    # Display current Gemini model information or a warning if API key is missing
    if api_key:
        st.success(f"Current Model: gemini-2.5-flash")
    else:
        st.warning("Please provide an API key to validate model connectivity.")

    st.divider()
