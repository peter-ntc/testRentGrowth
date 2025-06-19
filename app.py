import streamlit as st
import pandas as pd
from capm_optimizer import run_capm_optimizer

def capm_optimization_page():
    st.image("townsendAI_logo_1.png", width=100)
    st.title("📈 CAPM Optimization")
    if st.button("Back to Home", type="primary"):
        st.session_state.page = "home"

    try:
        df_returns = pd.read_excel("capm input.xlsx", sheet_name=0, usecols="B:O", nrows=2)
        df_returns.index = ["Expected Return", "Volatility"]
        sectors = pd.read_excel("capm input.xlsx", sheet_name=0, usecols="B:O", nrows=1, header=None).values.flatten()
        df_returns.columns = sectors
        df_returns = df_returns.T.reset_index()
        df_returns.columns = ["Sector", "Expected Return", "Volatility"]
        df_returns["Expected Return"] = df_returns["Expected Return"].map(lambda x: f"{x:.2%}")
        df_returns["Volatility"] = df_returns["Volatility"].map(lambda x: f"{x:.2%}")
        st.subheader("Expected Returns and Volatility")
        st.dataframe(df_returns, use_container_width=True)

        df_corr = pd.read_excel("capm input.xlsx", sheet_name=0, skiprows=7, usecols="B:O", nrows=14, header=None)
        df_corr.columns = sectors
        df_corr.index = sectors
        df_corr_display = df_corr.copy().applymap(lambda x: f"{x:.2%}")
        st.subheader("Correlation Matrix")
        st.dataframe(df_corr_display, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load or process 'capm input.xlsx'. Error: {e}")
        return

    if st.button("Run Optimization"):
        st.image("townsendAI_logo_1.png", width=80)
        st.header("Efficient Frontier")
        try:
            image_bytes, excel_bytes = run_capm_optimizer()
            st.image(image_bytes)
            st.download_button("Download CAPM Output (Excel)", data=excel_bytes,
                               file_name="capm_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error running optimizer: {e}")

if "page" not in st.session_state:
    st.session_state.page = "capm_optimization"

if st.session_state.page == "capm_optimization":
    capm_optimization_page()