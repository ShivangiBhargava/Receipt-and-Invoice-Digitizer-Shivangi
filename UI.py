#writefile app.py - https://clustered-debilitative-yuri.ngrok-free.dev/ (Live Demo Link)
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.chart_container import chart_container
from streamlit_extras.metric_cards import style_metric_cards

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
        hist_data = pd.DataFrame({'Date': ['12/29', '12/29', '12/29', '12/29'], 'Amount': [50, 80, 1000, 250]})
        st.bar_chart(hist_data, x="Date", y="Amount", color="#5d51e5")

# TAB 2: ANALYZE BILL
with tabs[1]:
    st.title("Analyze New Bill")
    st.write("Upload a receipt or invoice image to extract itemized data.")

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
  
