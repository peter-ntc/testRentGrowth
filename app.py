
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from capm_optimizer import run_capm_optimizer

def capm_optimization_page():
    st.image("townsendAI_logo_1.png", width=100)
    st.title("📈 CAPM Optimization")

    st.header("Efficient Frontier")

    try:
        # Load input file
        input_file = "capm_input.xlsx"
        sectors = pd.read_excel(input_file, sheet_name=0, header=2, nrows=1).T
        sectors.columns = ["Sector"]
        returns = pd.read_excel(input_file, sheet_name=0, header=3, nrows=1).T
        returns.columns = ["Expected Return"]
        volatility = pd.read_excel(input_file, sheet_name=0, header=4, nrows=1).T
        volatility.columns = ["Volatility"]
        corr_matrix = pd.read_excel(input_file, sheet_name=0, header=7, nrows=14, index_col=0)

        # Display tables
        summary_df = pd.concat([sectors, returns, volatility], axis=1).dropna()
        summary_df.index = range(len(summary_df))
        summary_df["Expected Return"] = (summary_df["Expected Return"]).map("{:.2%}".format)
        summary_df["Volatility"] = (summary_df["Volatility"]).map("{:.2%}".format)

        st.subheader("Expected Returns and Volatility")
        st.dataframe(summary_df)

        st.subheader("Correlation Matrix")
        corr_formatted = corr_matrix.applymap(lambda x: f"{x:.2%}")
        st.dataframe(corr_formatted)

        # Trigger optimizer
        if st.button("Run Optimization"):
            try:
                run_capm_optimizer(input_file=input_file)
                st.image("efficient_frontier.png", caption="Efficient Frontier", use_container_width=True)
                st.success("Optimization complete. Please check the downloaded Excel output.")
            except Exception as e:
                st.error(f"Error running optimizer: {e}")

    except Exception as e:
        st.error(f"Failed to load or process 'capm_input.xlsx'. Error: {e}")

def main():
    st.set_page_config(page_title="TownsendAI", layout="centered")
    page = st.sidebar.selectbox("Navigate", ["CAPM Optimization"])

    if page == "CAPM Optimization":
        capm_optimization_page()

if __name__ == "__main__":
    main()
