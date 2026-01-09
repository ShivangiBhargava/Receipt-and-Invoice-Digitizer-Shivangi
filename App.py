
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
from pdf2image import convert_from_bytes 
import plotly.express as px

DB_NAME = "receipts_vault.db"

def init_db():
  conn = sqlite3.connect(DB_NAME)
  c = conn.cursor()
  c.execute('''CREATE TABLE IF NOT EXISTS receipts
              (id INTEGER PRIMARY KEY AUTOINCREMENT,
               merchant TEXT,
               date TEXT,
               total REAL,
               currency TEXT,
               raw_json TEXT,
               timestamp DATETIME)''')

  conn.commit()
  conn.close()

def save_to_db(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    merchant = data.get('merchant', 'Unknown')
    date = data.get('date', 'Unknown')
    total = data.get('total', 0.0)
    currency = data.get('currency', '')
    items = json.dumps(data.get('items', []))

    c.execute('''INSERT INTO receipts (merchant, date, total, currency, raw_json, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)''',
             (merchant, date, total, currency, items, datetime.now()))
    conn.commit()
    conn.close()

def preprocess_image(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img = img[:, :, ::-1].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2 
    )
    return Image.fromarray(thresh) 

init_db()

st.set_page_config(page_title="Receipt and Invoice Digitizer", layout="wide", page_icon="🧾") 

with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    if st.button("Clear All Records"):
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
            init_db()
            st.rerun()

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

def analyze_receipt(image_data):
    prompt = """Extract receipt details into JSON:
    {
      "merchant": "string",
      "date": "string",
      "total": number,
      "currency": "string",
      "items": [{"name": "string", "qty": number, "price": number}]

}

Return ONLY JSON."""
    response = model.generate_content([prompt, image_data])
    # Clean up model's response which sometimes includes markdown fences
    clean_json = response.text.replace("```json", "").replace("```", "").strip() 
    return json.loads(clean_json)

# --- Main UI Logic with Tabs
st.title("🧾Receipt and Invoice Digitizer") 
tab1, tab2 = st.tabs(["Vault & Upload","Analytics Dashboard"])

with tab1:
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Upload Receipt (JPG/PNG/PDF)", type=["jpg", "png", "pdf"]) 

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                images = convert_from_bytes(uploaded_file.read())
                original_image = images[0]
            else:
                original_image = Image.open(uploaded_file)

            st.markdown("### Image Processing Comparison")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.image(original_image, caption="Original Image", use_container_width=True) 
            processed_image = preprocess_image(original_image)
            with comp_col2:
                st.image(processed_image, caption="Cleaned Image", use_container_width=True) 

            if st.button("Process & Save to Vault", use_container_width=True): 
                if not api_key:
                    st.error("Please enter your API Key in the sidebar.")
                else:
                    with st.spinner("Analyzing items ... "):
                        try:
                            extracted = analyze_receipt(processed_image)
                            save_to_db(extracted)
                            st.success(f"Stored {len(extracted.get('items', []))} items from {extracted.get('merchant', 'Unknown')}.") 
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

    with col2: 
        st.subheader("Persistent Storage")
        conn = sqlite3.connect(DB_NAME)
        history_df = pd.read_sql_query("SELECT * FROM receipts ORDER BY timestamp DESC", conn) 
        conn.close()

        if not history_df.empty:
            st.dataframe(history_df.drop(columns=['raw_json']), use_container_width=True) 
            st.markdown("### Detailed Bill Items")
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

with tab2:
    st.subheader("📊 Spending Insights") 
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM receipts", conn)
    conn.close()

    if not df.empty:
        # Data Cleaning for Analytics
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Dashboard Layout
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

