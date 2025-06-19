
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(page_title="TownsendAI", layout="centered")

def render_clickable_logo(image_path, url, width=120):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    html = f'''
        <div style="text-align:center;">
            <a href="{url}" target="_blank">
                <img src="data:image/png;base64,{img_str}" width="{width}" style="margin-bottom: 1rem;" />
            </a>
        </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def show_back_button():
    if st.button("🔙 Back to Home", key=f"back_{st.session_state.page}"):
        st.session_state.page = "home"
        st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "home"

render_clickable_logo("townsendAI_logo 1.png", "https://townsendgroup.com")

def home():
    st.title("🏢 TownsendAI Platform")
    st.markdown("### Explore our tools:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Forecasting & Modeling"):
            st.session_state.page = "forecasting"
            st.rerun()
        if st.button("🧠 Smart Benchmarks"):
            st.session_state.page = "benchmarks"
            st.rerun()
    with col2:
        if st.button("⚙️ Optimizer"):
            st.session_state.page = "optimizer"
            st.rerun()
        if st.button("📊 Secondaries Marketplace"):
            st.session_state.page = "secondaries"
            st.rerun()
    with col3:
        if st.button("💼 Fund & Deal Pipeline"):
            st.session_state.page = "pipeline"
            st.rerun()
        if st.button("📰 Market Research"):
            st.session_state.page = "research"
            st.rerun()

def coming_soon(section):
    st.title(section)
    show_back_button()
    st.info("🚧 This section is under construction. Stay tuned!")

if st.session_state.page == "home":
    home()
elif st.session_state.page == "optimizer":
    st.title("⚙️ Optimizer")
    show_back_button()
    st.markdown("### Choose an optimization tool:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📉 CAPM"):
            st.session_state.page = "capm"
            st.rerun()
    with col2:
        if st.button("📊 Sector Optimization"):
            st.session_state.page = "sector_opt"
            st.rerun()
    with col3:
        if st.button("💼 Fund Optimization"):
            st.session_state.page = "fund_opt"
            st.rerun()
elif st.session_state.page == "capm":
    st.title("📉 CAPM - Sector Inputs")
    show_back_button()
    file_path = "capm input.xlsx"
    try:
        capm_df = pd.read_excel(file_path, sheet_name=0, header=None)
        sectors = capm_df.iloc[2, 1:15].tolist()
        returns = capm_df.iloc[3, 1:15].tolist()
        vols = capm_df.iloc[4, 1:15].tolist()
        corr_matrix = capm_df.iloc[7:22, 0:15]
        corr_matrix.columns = ["Sector"] + sectors
        corr_matrix.set_index("Sector", inplace=True)

        st.subheader("Expected Returns and Volatility")
        df_summary = pd.DataFrame({
            "Sector": sectors,
            "Expected Return": returns,
            "Volatility": vols
        })
        st.dataframe(df_summary)

        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)

        if st.button("Run Optimization"):
            st.session_state.page = "capm_opt"
            st.rerun()

    except Exception as e:
        st.error(f"Failed to load 'capm input.xlsx'. Please upload it to the app folder. Error: {e}")
elif st.session_state.page == "capm_opt":
    coming_soon("📉 CAPM Optimization")
elif st.session_state.page == "sector_opt":
    coming_soon("📊 Sector Optimization")
elif st.session_state.page == "fund_opt":
    coming_soon("💼 Fund Optimization")
elif st.session_state.page == "pipeline":
    coming_soon("💼 Fund & Deal Pipeline")
elif st.session_state.page == "benchmarks":
    coming_soon("🧠 Smart Benchmarks")
elif st.session_state.page == "secondaries":
    coming_soon("📊 Secondaries Marketplace")
elif st.session_state.page == "research":
    coming_soon("📰 Market Research")
elif st.session_state.page == "forecasting":
    coming_soon("📈 Forecasting & Modeling")
