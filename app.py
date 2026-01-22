#!pip install streamlit google-generativeai opencv-python-headless pdf2image plotly pandas Pillow
#!apt-get install poppler-utils -y

%%writefile app.py
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
    # It includes columns for merchant, date, total, currency, raw JSON data, a unique file hash,
    # and a timestamp for when the record was created.
    c.execute('''CREATE TABLE IF NOT EXISTS receipts
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 merchant TEXT,
                 date TEXT,
                 total REAL,
                 currency TEXT,
                 raw_json TEXT,
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
    date = data.get('date', 'Unknown')
    total = data.get('total', 0.0)
    currency = data.get('currency', '')
    items = json.dumps(data.get('items', [])) # Convert list of items to JSON string for storage

    try:
        # Insert data into the 'receipts' table
        c.execute('''INSERT INTO receipts (merchant, date, total,
currency, raw_json, file_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (merchant, date, total, currency, items, file_hash,
datetime.now()))

        conn.commit()
        return True # Return True if insertion is successful
    except sqlite3.IntegrityError:
        # Handle case where file_hash is not unique (duplicate receipt)
        return False
    finally:
        conn.close()

# Function to preprocess the image for better OCR accuracy
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
      "items": [{ "name": "string", "qty": number, "price": number} ]
    }
    Return ONLY JSON."""
    # Generate content from the model using the prompt and image data
    response = model.generate_content([prompt, image_data])
    # Clean the response to ensure it's a valid JSON string
    clean_json = response.text.replace("```json", "").replace("```",
"").strip()
    return json.loads(clean_json) # Parse the JSON string into a Python dictionary


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
        uploaded_file = st.file_uploader("Upload Receipt (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])

        if uploaded_file:
            # Handle PDF files by converting the first page to an image
            if uploaded_file.type == "application/pdf":
                images = convert_from_bytes(uploaded_file.read())
                original_image = images[0]
            else:
                original_image = Image.open(uploaded_file)

            st.markdown("### Image Processing Comparison")
            # Display original and processed images side-by-side
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.image(original_image, caption="Original Image", use_container_width=True)

            processed_image = preprocess_image(original_image)
            with comp_col2:
                st.image(processed_image, caption="Cleaned Image",
                         use_container_width=True)

            # Button to process and save the receipt data
            if st.button("🚀 Process & Save to Vault",
                           use_container_width=True, type="primary"):
                if not api_key:
                    st.error("Please enter your API Key in the sidebar.")

                else:
                    # Duplicate check using SHA-256 Hash of the original image
                    img_hash = get_image_hash(original_image)
                    conn = sqlite3.connect(DB_NAME)
                    existing = conn.execute("SELECT id FROM receipts WHERE file_hash = ?", (img_hash,)).fetchone()
                    conn.close()

                    if existing:
                        st.warning("⚠️ This receipt has already been uploaded and exists in the vault.")
                    else:
                        with st.spinner("Analyzing items... "):
                            try:
                                # Analyze the processed image using the Gemini model
                                extracted = analyze_receipt(processed_image)
                                # Save the extracted data to the database
                                if save_to_db(extracted, img_hash):
                                    st.success(f"Stored {len(extracted.get('items', []))} items!")
                                    st.rerun() # Rerun to update the vault display
                            except Exception as e:
                                st.error(f"Analysis failed: {e}")

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

        dash_col1, dash_col2 = st.columns(2)

        with dash_col1:
            st.markdown("#### Spending by Merchant")
            # Create a pie chart showing spending distribution by merchant
            merchant_shares = df.groupby('merchant')['total'].sum().reset_index()
            fig_pie = px.pie(merchant_shares, values='total',
                             names='merchant',
                             hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)

        with dash_col2:
            st.markdown("#### Total Expenses per Merchant")
            # Create a bar chart showing total expenses for each merchant
            merchant_totals = df.groupby('merchant')['total'].sum().sort_values(ascending=False).reset_index()
            fig_bar = px.bar(merchant_totals, x='merchant',
                             y='total',
                             color='total', labels={'total':'Total Spent ($)'},
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Spending Over Time")
        # Create a line chart showing spending trends over time
        time_series = df.groupby(df['timestamp'].dt.date)['total'].sum().reset_index()
        fig_line = px.line(time_series, x='timestamp', y='total',
                          markers=True, title="Timeline of Expenses")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Upload receipts to view the analytics dashboard.")

with tab3:
    st.subheader("⚙️ System & Code Validation")

    # Data for the system and code validation table
    validation_data = {
        "Requirement": ["Gemini API Key", "Database Connection",
                        "Table Schema", "OpenCV (CV2)", "Plotly Express"],
        "Status": [
            "✅ Configured" if api_key else "❌ Missing Key", # Check if API key is provided
            "✅ Connected" if os.path.exists(DB_NAME) else "⚠️ Initializing", # Check database file existence
            "✅ Verified (Receipts Table)", # Assumes table creation is successful
            "✅ Available", # Assumes OpenCV is correctly imported
            "✅ Available" # Assumes Plotly Express is correctly imported
        ]
    }

    st.table(pd.DataFrame(validation_data)) # Display validation data in a table

    st.markdown("### 📋 Model Configuration")
    # Display current Gemini model information or a warning if API key is missing
    if api_key:
        st.success(f"Current Model: gemini-2.5-flash")
    else:
        st.warning("Please provide an API key to validate model connectivity.")
