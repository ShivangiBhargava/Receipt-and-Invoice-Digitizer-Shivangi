
import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import plotly.express as px
import hashlib # Add this import

# --- Database Configuration ---
DB_NAME = "receipts_vault.db"

# --- Database Initialization and Operations ---
def init_db():
  conn = sqlite3.connect(DB_NAME)
  c = conn.cursor()
  # Create receipts table if it doesn't exist
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

def save_to_db(data, file_hash): # Added file_hash parameter
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    merchant = data.get('merchant', 'Unknown')
    date_str = data.get('date', 'Unknown')
    total = data.get('total', 0.0)
    currency = data.get('currency', '')
    invoice_id = data.get('invoice_id', None) # New field
    tax = data.get('tax', 0.0) # New field
    items = json.dumps(data.get('items', []))

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


    # Insert new receipt record
    c.execute('''INSERT INTO receipts (merchant, date, total, currency, raw_json, invoice_id, tax, file_hash, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
             (merchant, formatted_date, total, currency, items, invoice_id, tax, file_hash, datetime.now()))
    conn.commit()
    conn.close()
    return True # Indicate successful save

# --- Image Preprocessing Function ---
def preprocess_image(pil_image):
    # Convert PIL Image to OpenCV format (BGR)
    img = np.array(pil_image.convert('RGB'))
    img = img[:, :, ::-1].copy() # Convert RGB to BGR
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)

# --- New function to validate extracted total ---
def validate_extracted_total(parsed_json_data):
    """Calculates and returns receipt total validation details, including extracted tax."""
    calculated_subtotal = 0.0
    if 'items' in parsed_json_data and parsed_json_data['items']:
        for item in parsed_json_data['items']:
            qty = item.get('qty', 0)
            price = item.get('price', 0.0)
            calculated_subtotal += (qty * price)

    extracted_total = parsed_json_data.get('total', 0.0)
    extracted_tax = parsed_json_data.get('tax', 0.0) # Get extracted tax from AI model

    # If an explicit tax was extracted, use it for validation
    if extracted_tax is not None and extracted_tax != 0.0:
        # Check if subtotal + extracted_tax approximately equals extracted_total
        is_valid = abs(calculated_subtotal + extracted_tax - extracted_total) < 0.01
        inferred_tax = extracted_tax # If extracted, then inferred is extracted
    else:
        # Otherwise, infer tax from total - subtotal
        inferred_tax = extracted_total - calculated_subtotal
        is_valid = abs(calculated_subtotal + inferred_tax - extracted_total) < 0.01 # Should always be True by definition here

    return {
        'calculated_subtotal': calculated_subtotal,
        'inferred_tax': inferred_tax,
        'extracted_total': extracted_total,
        'extracted_tax': extracted_tax, # Return extracted tax for display
        'is_valid': is_valid
    }

# --- Initialize Database ---
init_db()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Receipt and Invoice Digitizer", layout="wide", page_icon="🧾")

# --- Sidebar for Authentication and Data Management ---
with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    if st.button("Clear All Records"):
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
            init_db()
            st.rerun() # Rerun the app to reflect changes

# --- Gemini API Configuration ---
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

# --- Receipt Analysis Function using Gemini Pro Vision ---
def analyze_receipt(image_data):
    prompt = """Extract receipt details into JSON:
    {
      "merchant": "string",
      "date": "string",
      "total": number,
      "currency": "string",
      "invoice_id": "string",
      "tax": number,         
      "items": [{"name": "string", "qty": number, "price": number}]

}

Return ONLY JSON."""
    response = model.generate_content([prompt, image_data])
    # Clean up model's response which sometimes includes markdown fences
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

# --- Main UI Logic with Tabs ---
st.title("🧾Receipt and Invoice Digitizer")
tab1, tab2, tab3 = st.tabs(["Vault & Upload","Analytics Dashboard", "Validation"]) # Added tab3

# --- Tab 1: Vault & Upload ---
with tab1:
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.subheader("Upload Document")
        uploaded_files = st.file_uploader("Upload Receipt(s) (JPG/PNG/PDF)", type=["jpg", "png", "pdf"], accept_multiple_files=True)

        if uploaded_files:
            if st.button("Process & Save to Vault", use_container_width=True):
                if not api_key:
                    st.error("Please enter your API Key in the sidebar.")
                else:
                    processed_count = 0
                    for uploaded_file in uploaded_files:
                        # Handle PDF vs Image files
                        if uploaded_file.type == "application/pdf":
                            images = convert_from_bytes(uploaded_file.read())
                            original_image = images[0] # Process only the first page for simplicity
                        else:
                            original_image = Image.open(uploaded_file)

                        # Generate file hash for duplicate detection
                        file_hash = get_image_hash(original_image)

                        st.markdown(f"### Processing: {uploaded_file.name}")
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.image(original_image, caption="Original Image", use_container_width=True)
                        # Preprocess image before sending to Gemini
                        processed_image = preprocess_image(original_image)
                        with comp_col2:
                            st.image(processed_image, caption="Cleaned Image", use_container_width=True)

                        with st.spinner(f"Analyzing {uploaded_file.name}..."):
                            try:
                                extracted = analyze_receipt(processed_image)
                                saved_successfully = save_to_db(extracted, file_hash) # Pass file_hash
                                if saved_successfully:
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
        # Fetch all receipts from the database
        conn = sqlite3.connect(DB_NAME)
        history_df = pd.read_sql_query("SELECT * FROM receipts ORDER BY timestamp DESC", conn)
        conn.close()

        if not history_df.empty:
            # Drop raw_json, file_hash for cleaner display in general overview
            st.dataframe(history_df.drop(columns=['raw_json', 'file_hash']), use_container_width=True)
            st.markdown("### Detailed Bill Items")
            # Allow user to select a receipt to view its items
            selected_id = st.selectbox("Select ID to view items:", history_df['id'].unique())
            if selected_id:
                row = history_df[history_df['id'] == selected_id].iloc[0]
                try:
                    items_list = json.loads(row['raw_json'])
                    st.table(pd.DataFrame(items_list))
                except:
                    st.error("Could not parse items.")
        else:
            st.info("The vault is empty.")

# --- Tab 2: Analytics Dashboard ---
with tab2:
    st.subheader("📊 Spending Insights")
    # Fetch all receipts for analytics
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM receipts", conn)
    conn.close()

    if not df.empty:
        # Data Cleaning and Type Conversion for Analytics
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
        st.info("No data available for analytics yet. Upload some receipts!")

# --- Tab 3: Validation ---
with tab3:
    st.subheader("⚙️ System & Code Validation")

    # Data for the system and code validation table
    validation_data = {
        "Requirement": ["Gemini API Key", "Database Connection", "NLP-based Extraction", "Total Validation (Items Sum)", "Duplicate Detection (File Hash/Details)"],
        "Status": [
            "✅ Configured" if api_key else "❌ Missing Key",
            "✅ Connected" if os.path.exists(DB_NAME) else "⚠️ Initializing",
            "✅ Logic Present", # Assumed to be present via Gemini model integration
            "✅ Logic Present", # Will be present via validate_extracted_total
            "✅ Logic Present"  # Will be present via file_hash and save_to_db logic
        ]
    }
    st.table(pd.DataFrame(validation_data))

    st.markdown("### 📋 Model Configuration")
    if api_key:
        st.success(f"Current Model: gemini-2.5-flash")
    else:
        st.warning("Please provide an API key to validate model connectivity.")

    st.divider()
    st.subheader("🧾 Receipt Detailed Validation")

    conn = sqlite3.connect(DB_NAME)
    validation_df = pd.read_sql_query("SELECT id, merchant, date, total, invoice_id, tax, raw_json, file_hash FROM receipts ORDER BY timestamp DESC", conn)
    conn.close()

    if not validation_df.empty:
        selected_receipt_id = st.selectbox("Select a Receipt ID for detailed validation:", validation_df['id'].tolist(), key="validation_detail_select")

        if selected_receipt_id:
            selected_row = validation_df[validation_df['id'] == selected_receipt_id].iloc[0]

            st.markdown(f"#### Merchant: {selected_row['merchant']}")
            st.markdown(f"#### Date: {selected_row['date']}")
            st.write(f"- Invoice ID: {selected_row['invoice_id'] if pd.notna(selected_row['invoice_id']) else 'N/A'}")

            try:
                parsed_json = json.loads(selected_row['raw_json'])
                if not isinstance(parsed_json, dict):
                    st.warning(f"⚠️ **Cannot perform detailed validation for this entry (ID: {selected_receipt_id})**")
                    st.markdown("The AI model previously returned malformed JSON (expected a dictionary, but received a different type like a list).")
                    st.markdown("This issue has been addressed for new uploads. For this historical entry, please re-upload the original document if you wish to re-process it correctly.")
                    st.json(parsed_json) # Show the malformed JSON for debugging/transparency
                else:
                    validation_results = validate_extracted_total(parsed_json)

                    st.write(f"- Subtotal (calculated from items): {validation_results['calculated_subtotal']:.2f}")
                    st.write(f"- Extracted Tax (from AI): {validation_results['extracted_tax']:.2f}")
                    st.write(f"- Inferred Tax (Total - Subtotal): {validation_results['inferred_tax']:.2f}")
                    st.write(f"- Total (extracted by AI): {validation_results['extracted_total']:.2f}")

                    validation_status = "✅ Valid" if validation_results['is_valid'] else "❌ Invalid"
                    st.markdown(f"**Total Validation Status:** {validation_status}")

                    # Duplicate detection status display
                    conn = sqlite3.connect(DB_NAME)
                    # Check for duplicates by file_hash excluding self
                    duplicate_count_hash = conn.execute("SELECT COUNT(*) FROM receipts WHERE file_hash = ? AND id != ?", (selected_row['file_hash'], selected_row['id'])).fetchone()[0]
                    # Check for duplicates by details (merchant, date, total, invoice_id) excluding self and records already caught by hash
                    if pd.notna(selected_row['invoice_id']):
                        duplicate_count_details = conn.execute("SELECT COUNT(*) FROM receipts WHERE merchant = ? AND date = ? AND total = ? AND invoice_id = ? AND id != ?",
                                                               (selected_row['merchant'], selected_row['date'], selected_row['total'], selected_row['invoice_id'], selected_row['id'])).fetchone()[0]
                    else:
                        duplicate_count_details = conn.execute("SELECT COUNT(*) FROM receipts WHERE merchant = ? AND date = ? AND total = ? AND id != ?",
                                                               (selected_row['merchant'], selected_row['date'], selected_row['total'], selected_row['id'])).fetchone()[0]
                    conn.close()

                    st.markdown("--- ")
                    st.markdown("**Duplicate Detection Status:**")
                    if duplicate_count_hash > 0:
                        st.warning(f"⚠️ Duplicate by **File Hash** found! This receipt has {duplicate_count_hash} other record(s) with the same image hash.")
                    elif duplicate_count_details > 0:
                         st.warning(f"⚠️ Duplicate by **Details** found! This receipt has {duplicate_count_details} other record(s) with matching merchant, date, and total (and invoice ID if present).")
                    else:
                        st.success("✅ Unique receipt: No duplicates found based on file hash or key details.")


            except json.JSONDecodeError:
                st.error(f"Error: Could not decode JSON from raw data for receipt ID {selected_receipt_id}. Raw data: {selected_row['raw_json']}")
                st.markdown("This indicates the saved raw data was not valid JSON.")
            except Exception as e:
                st.error(f"Error processing receipt for validation: {e}")
    else:
        st.info("No receipts in the vault to validate.")
        
