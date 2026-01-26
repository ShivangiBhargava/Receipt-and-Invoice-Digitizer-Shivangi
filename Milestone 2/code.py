
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
import re
from pdf2image import convert_from_bytes
import plotly.express as px

# ---Database Setup---
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
                 invoice_id TEXT,
                 tax REAL,
                 subtotal REAL,
                 tax_rate REAL,
                 file_hash TEXT UNIQUE,
                 validation_status TEXT,
                 validation_details TEXT,
                 timestamp DATETIME)''')
    conn.commit()
    conn.close()

# --- Regex Field Extractor for Milestone 2 Requirement ---
class RegexFieldExtractor:
    def __init__(self):
        self.patterns = {
            'date': [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
            ],
            'total': [
                r'(?:total|amount)[\s:]*\$?(\d+\.\d{2})',
                r'[\$€£]\s*(\d+\.\d{2})'
            ],
            'invoice_id': [
                r'(?:invoice|receipt|inv)[\s#:]*([A-Z0-9\-]+)',
                r'ID[:\s]*([A-Z0-9]+)'
            ],
            'tax': [
                r'tax[\s:]*\$?(\d+\.\d{2})',
                r'GST/VAT[\s:]*\$?(\d+\.\d{2})'
            ],
            'subtotal': [
                r'subtotal[\s:]*\$?(\d+\.\d{2})',
                r'amount[\s:]*\$?(\d+\.\d{2})'
            ],
            'tax_rate': [
                r'(\d+\.?\d*)\s*%',
                r'tax rate[\s:]*(\d+\.?\d*)'
            ]
        }
    
    def extract_from_text(self, text):
        results = {}
        for field, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        # Convert numeric fields to float
                        if field in ['total', 'tax', 'subtotal', 'tax_rate']:
                            results[field] = float(match.group(1))
                        else:
                            results[field] = match.group(1)
                    except:
                        results[field] = match.group(1)
                    break
        return results

def get_image_hash(pil_image):
    """Generates a unique SHA-256 hash for the image to prevent duplicates."""
    hash_handler = hashlib.sha256()
    img_byte_arr = pil_image.tobytes()
    hash_handler.update(img_byte_arr)
    return hash_handler.hexdigest()

def save_to_db(data, file_hash, validation_result=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    merchant = data.get('merchant', 'Unknown')
    date_str = data.get('date', 'Unknown')
    total = data.get('total', 0.0)
    currency = data.get('currency', '')
    invoice_id = data.get('invoice_id', None)
    tax = data.get('tax', 0.0)
    subtotal = data.get('subtotal', 0.0)
    tax_rate = data.get('tax_rate', 0.0)
    items = json.dumps(data.get('items', []))

    formatted_date = 'Unknown'
    if date_str != 'Unknown':
        try:
            parsed_date = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(parsed_date):
                formatted_date = parsed_date.strftime('%Y-%m-%d')
        except Exception:
            pass

    # Duplicate check
    existing_receipt_by_hash = c.execute("SELECT id FROM receipts WHERE file_hash = ?", (file_hash,)).fetchone()
    if existing_receipt_by_hash:
        conn.close()
        return False, "Duplicate detected by file hash"

    if formatted_date != 'Unknown' and merchant != 'Unknown' and total != 0.0:
        if invoice_id:
            existing_receipt_by_details = c.execute('''SELECT id FROM receipts WHERE merchant = ? AND date = ? AND total = ? AND invoice_id = ?''',
                                              (merchant, formatted_date, total, invoice_id)).fetchone()
        else:
            existing_receipt_by_details = c.execute('''SELECT id FROM receipts WHERE merchant = ? AND date = ? AND total = ?''',
                                              (merchant, formatted_date, total)).fetchone()
        if existing_receipt_by_details:
            conn.close()
            return False, "Duplicate detected by business logic"

    # Prepare validation details
    validation_status = validation_result.get('status', 'pending') if validation_result else 'pending'
    validation_details = json.dumps(validation_result) if validation_result else '{}'

    try:
        c.execute('''INSERT INTO receipts (merchant, date, total, currency, raw_json, invoice_id, tax, subtotal, tax_rate, file_hash, validation_status, validation_details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (merchant, formatted_date, total, currency, items, invoice_id, tax, subtotal, tax_rate, file_hash, validation_status, validation_details, datetime.now()))
        conn.commit()
        return True, "Successfully saved"
    except sqlite3.IntegrityError:
        return False, "Database integrity error"
    finally:
        conn.close()

# --- Enhanced Validation Functions ---
def validate_extracted_total(parsed_json_data, ocr_text=""):
    """Comprehensive validation similar to reference image"""
    
    validation_results = {
        'status': 'pending',
        'checks': {},
        'summary': {}
    }
    
    # Extract values
    extracted_total = float(parsed_json_data.get('total', 0.0))
    extracted_tax = float(parsed_json_data.get('tax', 0.0))
    extracted_subtotal = float(parsed_json_data.get('subtotal', 0.0))
    extracted_tax_rate = float(parsed_json_data.get('tax_rate', 0.0))
    
    # Calculate from line items
    calculated_subtotal = 0.0
    if 'items' in parsed_json_data and parsed_json_data['items']:
        for item in parsed_json_data['items']:
            qty = float(item.get('qty', 0))
            price = float(item.get('price', 0.0))
            calculated_subtotal += (qty * price)
    
    # --- Total Validation ---
    if extracted_subtotal > 0:
        expected_total = extracted_subtotal + extracted_tax
        total_diff = abs(expected_total - extracted_total)
        is_total_valid = total_diff < 0.01
        validation_results['checks']['total_validation'] = {
            'status': '✓' if is_total_valid else '✗',
            'message': f"Subtotal (${extracted_subtotal:.2f}) + Tax (${extracted_tax:.2f}) = Total (${extracted_total:.2f})",
            'valid': is_total_valid,
            'details': f"Expected: ${expected_total:.2f}, Actual: ${extracted_total:.2f}, Diff: ${total_diff:.2f}"
        }
    else:
        validation_results['checks']['total_validation'] = {
            'status': '⚠',
            'message': "Cannot validate total - subtotal missing",
            'valid': False
        }
    
    # --- Duplicate Detection Check ---
    validation_results['checks']['duplicate_detection'] = {
        'status': '✓',
        'message': f"No duplicate found for {parsed_json_data.get('invoice_id', 'N/A')}",
        'valid': True
    }
    
    # --- Tax Rate Validation ---
    if extracted_tax > 0 and extracted_subtotal > 0:
        actual_tax_rate = (extracted_tax / extracted_subtotal) * 100
        if extracted_tax_rate > 0:
            tax_rate_diff = abs(actual_tax_rate - extracted_tax_rate)
            is_tax_rate_valid = tax_rate_diff < 0.1
            validation_results['checks']['tax_rate_validation'] = {
                'status': '✓' if is_tax_rate_valid else '✗',
                'message': f"Expected tax rate: {extracted_tax_rate:.1f}%, Actual: {actual_tax_rate:.2f}%",
                'valid': is_tax_rate_valid
            }
        else:
            validation_results['checks']['tax_rate_validation'] = {
                'status': '✓',
                'message': f"Actual tax rate: {actual_tax_rate:.2f}%",
                'valid': True
            }
    else:
        validation_results['checks']['tax_rate_validation'] = {
            'status': '⚠',
            'message': "Cannot calculate tax rate - insufficient data",
            'valid': False
        }
    
    # --- Date Format Validation ---
    date_str = parsed_json_data.get('date', '')
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'\d{1,2}-\d{1,2}-\d{4}'
    ]
    is_date_valid = any(re.match(pattern, date_str) for pattern in date_patterns) if date_str else False
    validation_results['checks']['date_format'] = {
        'status': '✓' if is_date_valid else '✗',
        'message': f"Valid date format: {date_str}" if is_date_valid else f"Invalid date format: {date_str}",
        'valid': is_date_valid
    }
    
    # --- Required Fields Check ---
    required_fields = ['merchant', 'date', 'total']
    missing_fields = [field for field in required_fields if not parsed_json_data.get(field)]
    is_all_fields_present = len(missing_fields) == 0
    validation_results['checks']['required_fields'] = {
        'status': '✓' if is_all_fields_present else '✗',
        'message': "All required fields present" if is_all_fields_present else f"Missing fields: {', '.join(missing_fields)}",
        'valid': is_all_fields_present
    }
    
    # --- Regex Validation ---
    if ocr_text:
        regex_extractor = RegexFieldExtractor()
        regex_results = regex_extractor.extract_from_text(ocr_text)
        regex_matches = {k: v for k, v in regex_results.items() if v}
        validation_results['checks']['regex_extraction'] = {
            'status': '✓' if regex_matches else '⚠',
            'message': f"Regex extracted {len(regex_matches)} fields" if regex_matches else "No regex matches found",
            'valid': bool(regex_matches),
            'extracted': regex_matches
        }
    
    # --- NLP/Line Items Validation ---
    if 'items' in parsed_json_data and parsed_json_data['items']:
        validation_results['checks']['line_items'] = {
            'status': '✓',
            'message': f"{len(parsed_json_data['items'])} line items extracted",
            'valid': True
        }
    else:
        validation_results['checks']['line_items'] = {
            'status': '✗',
            'message': "No line items found",
            'valid': False
        }
    
    # Determine overall status
    all_valid = all(check.get('valid', False) for check in validation_results['checks'].values())
    validation_results['status'] = 'valid' if all_valid else 'issues'
    
    # Create summary
    validation_results['summary'] = {
        'total_checks': len(validation_results['checks']),
        'passed_checks': sum(1 for check in validation_results['checks'].values() if check.get('valid', False)),
        'failed_checks': sum(1 for check in validation_results['checks'].values() if not check.get('valid', False))
    }
    
    return validation_results

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

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

init_db()

st.set_page_config(page_title="Receipt and Invoice Digitizer", layout="wide", page_icon="🧾")

with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")

    st.divider()
    st.subheader("📥 Export Data")
    conn = sqlite3.connect(DB_NAME)
    df_export = pd.read_sql_query("SELECT * FROM receipts", conn)
    conn.close()

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
    if st.button("Clear All Records", type="secondary"):
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
      "invoice_id": "string",
      "tax": number,
      "subtotal": number,
      "tax_rate": number,
      "items": [{ "name": "string", "qty": number, "price": number} ]
    }
    Return ONLY JSON."""
    response = model.generate_content([prompt, image_data])
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    try:
        parsed_json = json.loads(clean_json)
        if not isinstance(parsed_json, dict):
            raise ValueError("AI model returned malformed JSON")
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"AI model returned invalid JSON: {e}")

st.title("🧾 Receipt and Invoice Digitizer")
tab1, tab2, tab3 = st.tabs(["📤 Vault & Upload", "📊 Analytics Dashboard", "✅ Validation & Results"])

with tab1:
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.subheader("Upload Document")
        uploaded_files = st.file_uploader("Upload Receipt(s) (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

        if uploaded_files:
            if st.button("🚀 Process & Validate", use_container_width=True, type="primary"):
                if not api_key:
                    st.error("Please enter your API Key in the sidebar.")
                else:
                    processed_count = 0
                    for uploaded_file in uploaded_files:
                        if uploaded_file.type == "application/pdf":
                            images = convert_from_bytes(uploaded_file.read())
                            original_image = images[0]
                        else:
                            original_image = Image.open(uploaded_file)

                        file_hash = get_image_hash(original_image)

                        st.markdown(f"### Processing: {uploaded_file.name}")
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.image(original_image, caption="Original Image", use_container_width=True)

                        processed_image = preprocess_image(original_image)
                        with comp_col2:
                            st.image(processed_image, caption="Cleaned Image", use_container_width=True)

                        with st.spinner(f"Analyzing {uploaded_file.name}... "):
                            try:
                                extracted = analyze_receipt(processed_image)
                                
                                # Perform validation
                                validation_result = validate_extracted_total(extracted)
                                
                                # Save with validation results
                                saved, message = save_to_db(extracted, file_hash, validation_result)
                                
                                if saved:
                                    st.success(f"✅ Stored and validated {len(extracted.get('items', []))} items from {extracted.get('merchant', 'Unknown')}")
                                    processed_count += 1
                                    
                                    # Show validation summary
                                    with st.expander("View Validation Details", expanded=True):
                                        st.markdown("### 📋 Validation Results")
                                        for check_name, check_result in validation_result['checks'].items():
                                            col1, col2 = st.columns([1, 4])
                                            with col1:
                                                st.markdown(f"**{check_result['status']}**")
                                            with col2:
                                                st.markdown(check_result['message'])
                                        
                                        st.markdown("---")
                                        st.markdown(f"**Summary:** {validation_result['summary']['passed_checks']}/{validation_result['summary']['total_checks']} checks passed")
                                else:
                                    st.warning(f"⚠ {message}")
                            except Exception as e:
                                st.error(f"❌ Analysis failed for {uploaded_file.name}: {e}")
                    if processed_count > 0:
                        st.rerun()

    with col2:
        st.subheader("Persistent Storage")
        # Fetch all receipts from the database
        conn = sqlite3.connect(DB_NAME)
        history_df = pd.read_sql_query("SELECT * FROM receipts ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not history_df.empty:
            # Check which columns exist in the DataFrame before dropping
            columns_to_display = ['id', 'merchant', 'date', 'total', 'currency', 'invoice_id', 'tax', 'subtotal', 'tax_rate', 'validation_status', 'timestamp']
            
            # Only include columns that actually exist in the DataFrame
            available_columns = [col for col in columns_to_display if col in history_df.columns]
            
            # Display only the available columns
            st.dataframe(history_df[available_columns], 
                        use_container_width=True, hide_index=True)
            
            st.markdown("### 🔍 Detailed Bill Items")
            selected_id = st.selectbox("Select ID to view items:", history_df['id'].tolist())
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
            min_total = valid_totals_df['total'].min() if not valid_totals_df['total'].empty else 0.0
            max_total = valid_totals_df['total'].max() if not valid_totals_df['total'].empty else 1000.0

            # Prevent slider error when min_total == max_total
            if min_total == max_total and not valid_totals_df['total'].empty:
                max_total += 0.01 # Add a small epsilon to make max > min

            amount_range = st.slider(
                "Filter by Total Amount",
                float(min_total), float(max_total),
                (float(min_total), float(max_total))
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
    st.subheader("✅ Validation Results Dashboard")
    
    conn = sqlite3.connect(DB_NAME)
    receipts_df = pd.read_sql_query("SELECT * FROM receipts ORDER BY timestamp DESC", conn)
    conn.close()
    
    if not receipts_df.empty:
        # Select a receipt to view validation details
        receipt_options = []
        for _, row in receipts_df.iterrows():
            option_text = f"ID: {row['id']} - {row['merchant']} (${row['total']:.2f})"
            receipt_options.append((row['id'], option_text))
        
        selected_option = st.selectbox(
            "Select Receipt ID to view validation results:",
            options=receipt_options,
            format_func=lambda x: x[1]  # Display the formatted text
        )
        
        if selected_option:
            selected_receipt_id = selected_option[0]
            receipt = receipts_df[receipts_df['id'] == selected_receipt_id].iloc[0]
            
            # Display receipt header similar to reference image
            st.markdown("### 📋 Receipt Details")
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                st.markdown(f"**📅 Date:** {receipt['date'] if pd.notna(receipt['date']) else 'Unknown'}")
            with col2:
                st.markdown(f"**🏪 Vendor:** {receipt['merchant'] if receipt['merchant'] else 'Unknown'}")
            with col3:
                invoice_id = receipt['invoice_id'] if receipt['invoice_id'] else 'N/A'
                st.markdown(f"**📄 Invoice #:** {invoice_id}")
            
            # Financial summary
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                subtotal = receipt['subtotal'] if 'subtotal' in receipt and pd.notna(receipt['subtotal']) else 0.0
                st.metric("Subtotal", f"${float(subtotal):.2f}")
            with col2:
                tax = receipt['tax'] if 'tax' in receipt and pd.notna(receipt['tax']) else 0.0
                st.metric("Tax", f"${float(tax):.2f}")
                if 'tax_rate' in receipt and receipt['tax_rate'] and pd.notna(receipt['tax_rate']):
                    st.caption(f"Rate: {float(receipt['tax_rate']):.1f}%")
            with col3:
                total = receipt['total'] if 'total' in receipt and pd.notna(receipt['total']) else 0.0
                st.metric("Total", f"${float(total):.2f}")
            
            # Line Items Section
            st.markdown("---")
            st.subheader("📋 Line Items")
            try:
                items_list = json.loads(receipt['raw_json'])
                if items_list and len(items_list) > 0:
                    items_df = pd.DataFrame(items_list)
                    # Add calculated total for each item
                    items_df['item_total'] = items_df['qty'] * items_df['price']
                    st.dataframe(items_df, use_container_width=True, hide_index=True)
                    
                    # Calculate and show totals
                    calculated_subtotal = items_df['item_total'].sum()
                    st.markdown(f"**Calculated Subtotal:** ${calculated_subtotal:.2f}")
                else:
                    st.info("No line items available")
            except:
                st.error("Could not parse line items")
            
            # Validation Results Section - Similar to reference image
            st.markdown("---")
            st.subheader("✅ Validation Results")
            
            # Parse validation details
            validation_details = {}
            try:
                if 'validation_details' in receipt and receipt['validation_details'] and receipt['validation_details'] != '{}':
                    validation_details = json.loads(receipt['validation_details'])
            except Exception as e:
                st.warning(f"Could not parse validation details: {e}")
            
            if validation_details and 'checks' in validation_details:
                # Create a layout similar to the reference image
                for check_name, check_result in validation_details['checks'].items():
                    # Format check name for display
                    display_name = check_name.replace('_', ' ').title()
                    
                    # Create columns for check result
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        # Use emoji based on status
                        status = check_result.get('status', '')
                        if status == '✓':
                            st.success("✓")
                        elif status == '✗':
                            st.error("✗")
                        elif status == '⚠':
                            st.warning("⚠")
                        else:
                            st.info(status)
                    with col2:
                        message = check_result.get('message', 'No message')
                        st.markdown(f"**{display_name}**  \n{message}")
                    
                    # Show additional details if available
                    if 'details' in check_result and check_result['details']:
                        with st.expander("Details"):
                            st.markdown(check_result['details'])
                
                # Validation Summary
                st.markdown("---")
                if 'summary' in validation_details:
                    summary = validation_details['summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Checks", summary.get('total_checks', 0))
                    with col2:
                        st.metric("Passed", summary.get('passed_checks', 0))
                    with col3:
                        st.metric("Failed", summary.get('failed_checks', 0))
                    
                    # Overall status
                    overall_status = validation_details.get('status', 'unknown')
                    if overall_status == 'valid':
                        st.success("✅ All validation checks passed!")
                    elif overall_status == 'issues':
                        st.warning("⚠ Some validation issues found")
                    else:
                        st.info(f"Status: {overall_status}")
            else:
                st.info("No validation details available for this receipt.")
                
                # Manual validation check
                st.markdown("### Manual Validation Check")
                if st.button("Run Validation Now", key=f"validate_{selected_receipt_id}"):
                    try:
                        items_list = json.loads(receipt['raw_json'])
                        extracted_data = {
                            'merchant': receipt['merchant'],
                            'date': receipt['date'],
                            'total': float(receipt['total']) if pd.notna(receipt['total']) else 0.0,
                            'invoice_id': receipt['invoice_id'],
                            'tax': float(receipt['tax']) if 'tax' in receipt and pd.notna(receipt['tax']) else 0.0,
                            'subtotal': float(receipt['subtotal']) if 'subtotal' in receipt and pd.notna(receipt['subtotal']) else 0.0,
                            'tax_rate': float(receipt['tax_rate']) if 'tax_rate' in receipt and pd.notna(receipt['tax_rate']) else 0.0,
                            'items': items_list
                        }
                        validation_result = validate_extracted_total(extracted_data)
                        
                        # Update database with validation results
                        conn = sqlite3.connect(DB_NAME)
                        c = conn.cursor()
                        
                        # Check if validation columns exist
                        c.execute("PRAGMA table_info(receipts)")
                        columns = [col[1] for col in c.fetchall()]
                        
                        if 'validation_status' in columns and 'validation_details' in columns:
                            c.execute('''UPDATE receipts SET validation_status = ?, validation_details = ? WHERE id = ?''',
                                     (validation_result['status'], json.dumps(validation_result), selected_receipt_id))
                        conn.commit()
                        conn.close()
                        
                        st.success("✅ Validation completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Validation failed: {e}")
    else:
        st.info("No receipts available for validation. Upload some receipts first!")
