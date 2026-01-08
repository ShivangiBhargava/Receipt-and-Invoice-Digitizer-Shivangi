#writefile app.py - https://clustered-debilitative-yuri.ngrok-free.dev/ (Live Demo Link)
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.chart_container import chart_container
from streamlit_extras.metric_cards import style_metric_cards
import base64
import cv2 # Import OpenCV for image processing
from PIL import Image # Import Pillow for image handling
import fitz # Import PyMuPDF for PDF processing
import pytesseract # Import pytesseract for OCR

# --- Tesseract OCR Configuration ---
# Ensure tesseract is in your PATH or specify the full path to the executable
# For Colab, it's usually in /usr/bin/tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Page configuration
st.set_page_config(page_title="Receipt and Invoice Digitizer", layout="wide", initial_sidebar_state="collapsed")

# --- CSS Styling ---
#Page configuration
st.set_page_config(page_title="Bill Analyzer", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
    <style>
    .main { background-color: #f8f9fc; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #4e46e5; }
    .stButton>button {
        background-color: #5d51e5;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
    .upload-box {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background-color: white;
    }
    .savings-advice {
        background-color: #eef2ff;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #5d51e5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
col_logo, col_nav = st.columns([1, 2])
with col_logo:
    st.subheader("Receipt and Invoice Digitizer")

with col_nav:
    tabs = st.tabs([" Dashboard", " Analyze Bill", " History"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.title("Financial Overview")
    st.caption("Detailed breakdown of your tracked spending and categories.")


col_logo, col_nav = st.columns([1, 2])
with col_logo:
    st.subheader(" Bill Analyzer")
    
with col_nav:
    tabs = st.tabs([" Dashboard", " Analyze Bill", " History"])

# TAB 1: DASHBOARD
with tabs[0]:
    st.title("Financial Overview")
    st.caption("Detailed breakdown of your tracked spending and categories.")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Tracked Spending", "$1289.48", "Across 4 bills")
    m2.metric("Average Bill Value", "$322.37", "Last updated just now")
    m3.metric("Top Category", "Shopping", delta_color="off")
    style_metric_cards(border_left_color="#5d51e5")

    
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Spending by Category")
        chart_data = pd.DataFrame({'Category': ['Dining', 'Health', 'Shopping'], 'Value': [10, 20, 70]})
        st.vega_lite_chart(chart_data, {
            'mark': {'type': 'arc', 'innerRadius': 50},
            'encoding': {
                'theta': {'field': 'Value', 'type': 'quantitative'},
                'color': {'field': 'Category', 'type': 'nominal', 'scale': {'range': ['#f59e0b', '#10b981', '#5d51e5']}}
            }
        }, use_container_width=True)

    with c2:
        st.write("### Recent Spending History")
        hist_data = pd.DataFrame({'Date': ['01/03', '01/04', '01/05', '01/06'], 'Amount': [50, 80, 110, 235]})
        st.line_chart(hist_data, x="Date", y="Amount", color="#5d51e5", use_container_width=True)

# --- OCR Function ---
def perform_ocr(image_array):
    """
    Performs OCR on the given image array using Tesseract.
    Expected input: OpenCV image (numpy array).
    """
    if image_array is None:
        return "No image provided for OCR."
    try:
        # Convert the OpenCV image (numpy array) to a PIL Image
        # If it's a grayscale image, ensure PIL.Image.fromarray handles it correctly
        if len(image_array.shape) == 2: # Grayscale
            pil_image = Image.fromarray(image_array, mode='L')
        else: # Color image
            pil_image = Image.fromarray(image_array)

        text = pytesseract.image_to_string(pil_image)
        return text
    except Exception as e:
        return f"Error during OCR: {e}"


# --- TAB 2: ANALYZE BILL ---
        hist_data = pd.DataFrame({'Date': ['12/29', '12/29', '12/29', '12/29'], 'Amount': [50, 80, 1000, 250]})
        st.bar_chart(hist_data, x="Date", y="Amount", color="#5d51e5")

# TAB 2: ANALYZE BILL
with tabs[1]:
    st.title("Analyze New Bill")
    st.write("Upload a receipt or invoice image to extract itemized data.")

    # Initialize session state for storing the image to be OCR'd
    if 'image_to_ocr' not in st.session_state:
        st.session_state.image_to_ocr = None
    if 'ocr_output' not in st.session_state:
        st.session_state.ocr_output = ""
    if 'last_uploaded_file_name' not in st.session_state:
        st.session_state.last_uploaded_file_name = ""

    with st.container():
        uploaded_files = st.file_uploader("Drop image here", type=['jpg', 'png', 'pdf'], accept_multiple_files=True)
        preprocess_option = st.checkbox("Apply image preprocessing (Grayscale + Adaptive Threshold)", value=True)

        current_image_for_ocr = None

        if uploaded_files is not None:
            # For simplicity, we'll process the last uploaded file in the list for OCR if multiple are uploaded
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                st.subheader(f"Analyzing: {uploaded_file.name}")
                col_orig, col_proc = st.columns(2)

                if file_type.startswith('image/'):
                    image_bytes = uploaded_file.getvalue()
                    np_array = np.frombuffer(image_bytes, np.uint8)
                    original_image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                    with col_orig:
                        st.write("#### Original Image")
                        st.image(image_bytes, caption="Original", width='stretch')

                    if preprocess_option:
                        with col_proc:
                            st.write("#### Processed Image")
                            gray_image = cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2GRAY)
                            processed_image_cv2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                            processed_image_pil = Image.fromarray(processed_image_cv2)
                            st.image(processed_image_pil, caption="Processed (Grayscale + Adaptive Threshold)", width='stretch')
                            current_image_for_ocr = processed_image_cv2 # Store processed for OCR
                    else:
                        with col_proc:
                            st.write("#### No Preprocessing Applied")
                            st.info("Check 'Apply image preprocessing' to see the processed image.")
                            current_image_for_ocr = original_image_cv2 # Store original for OCR (if no processing)

                elif file_type == 'application/pdf':
                    try:
                        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                        page = doc.load_page(0)
                        pix = page.get_pixmap()
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                        if pix.n == 4:
                            original_image_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                        elif pix.n == 3:
                            original_image_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        else:
                            original_image_cv2 = img_array

                        with col_orig:
                            st.write("#### Original Image (First Page of PDF)")
                            st.image(original_image_cv2, caption="Original PDF Page", width='stretch', channels="BGR")

                        if preprocess_option:
                            with col_proc:
                                st.write("#### Processed Image (First Page of PDF)")
                                if len(original_image_cv2.shape) == 3:
                                    gray_image = cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2GRAY)
                                else:
                                    gray_image = original_image_cv2
                                processed_image_cv2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                                processed_image_pil = Image.fromarray(processed_image_cv2)
                                st.image(processed_image_pil, caption="Processed (Grayscale + Adaptive Threshold)", width='stretch')
                                current_image_for_ocr = processed_image_cv2 # Store processed for OCR
                        else:
                            with col_proc:
                                st.write("#### No Preprocessing Applied")
                                st.info("Check 'Apply image preprocessing' to see the processed image.")
                                current_image_for_ocr = original_image_cv2 # Store original for OCR (if no processing)
                        doc.close()
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                        st.warning("Could not render PDF preview. Displaying as document.")
                        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.warning(f"Unsupported file type uploaded: {uploaded_file.name}")

                # After processing each file, update the session state with the image to be OCR'd
                # This means OCR will run on the last uploaded/processed file if multiple are selected
                st.session_state.image_to_ocr = current_image_for_ocr
                st.session_state.last_uploaded_file_name = uploaded_file.name # Store name for display

        # "Extract Data & Analyze" button and OCR display
        if st.button("Extract Data & Analyze"):
            if st.session_state.image_to_ocr is not None:
                with st.spinner('Extracting text...'):
                    st.session_state.ocr_output = perform_ocr(st.session_state.image_to_ocr)
                st.success(f"OCR performed on {st.session_state.last_uploaded_file_name} successfully!")
            else:
                st.warning("Please upload an image or PDF first to perform OCR.")
                st.session_state.ocr_output = "" # Clear previous output if no image

        if st.session_state.ocr_output:
            st.write("### Extracted Text (Raw OCR Output)")
            st.text_area("OCR Result", st.session_state.ocr_output, height=300)

    st.info("**Pro Tip:** Make sure the lighting is good and the text is clearly visible.")

# --- TAB 3: HISTORY ---
    # upload UI
    with st.container():
        uploaded_files = st.file_uploader("Drop image here", type=['jpg', 'png', 'pdf'], accept_multiple_files=True)
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                if file_type.startswith('image/'):
                    st.image(uploaded_file, caption=f"Uploaded Bill: {uploaded_file.name}", width=300)
                elif file_type == 'application/pdf':
                    # Display PDF using an iframe with base64 encoding
                    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    st.success(f"PDF file '{uploaded_file.name}' uploaded successfully!")
                else:
                    st.warning(f"Unsupported file type uploaded: {uploaded_file.name}")

        st.button("Extract Data & Analyze")

    st.info("**Pro Tip:** Make sure the lighting is good and the text is clearly visible.")

# TAB 3: HISTORY
with tabs[2]:
    h_col1, h_col2 = st.columns([1, 2])

    with h_col1:
        st.write("### Past Bills")
        st.button("FRIES & CHIPS REST... \n $31.12")
        st.button("Burrito Bar \n $22.11")

    with h_col2:
        st.write("### LINE ITEMS")
        items = {
            "2 FRIES & CHIPS BOWL": "$17.98",
            "2 DRINKS": "$3.98",
            "2 FRESH SPICY SOUP": "$7.96",
            "TAX": "$1.20"
        }
        for item, price in items.items():
            st.write(f"**{item}** : {price}")

        st.markdown("""
        <div class="savings-advice">
            <p><strong> AI Savings Advice</strong></p>
            <i>"To save money on future visits, consider opting for water instead of paid drinks..."</i>
        </div>
        """, unsafe_allow_html=True)
            <p><strong> AI Savings Advice</strong></p>  
            <i>"To save money on future visits, consider opting for water instead of paid drinks..."</i> 
        </div>  
        """, unsafe_allow_html=True)
  
