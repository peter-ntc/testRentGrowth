
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# Page configuration
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>TownsendAI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Powered by Townsend's proprietary data and analytics platform</h3>", unsafe_allow_html=True)
st.markdown("---")

# Define logo if available locally (optional)
# st.image("townsendAI_logo_1.png", width=100)

# Navigation choices
pages = {
    "Forecasting & Modeling": "forecasting",
    "Optimizer": "optimizer",
    "Fund & Deal Pipeline": "coming_soon",
    "Smart Benchmarks": "coming_soon",
    "Secondaries Marketplace": "coming_soon",
    "Market Research": "coming_soon"
}

selection = st.selectbox("Select a module", [""] + list(pages.keys()))

if selection == "Forecasting & Modeling":
    st.info("Forecasting & Modeling module is under construction.")
elif selection == "Fund & Deal Pipeline":
    st.warning("Under construction. Coming soon.")
elif selection == "Smart Benchmarks":
    st.warning("Under construction. Coming soon.")
elif selection == "Secondaries Marketplace":
    st.warning("Under construction. Coming soon.")
elif selection == "Market Research":
    st.warning("Under construction. Coming soon.")
elif selection == "Optimizer":
    st.header("📈 CAPM Optimization")
    st.markdown("Upload the CAPM input Excel file.")

    uploaded_file = st.file_uploader("Upload capm_input.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df_returns = pd.read_excel(uploaded_file, sheet_name=0, header=None, usecols="B:O", nrows=1)
            df_volatility = pd.read_excel(uploaded_file, sheet_name=0, header=None, usecols="B:O", skiprows=1, nrows=1)
            df_corr = pd.read_excel(uploaded_file, sheet_name=0, header=None, usecols="A:O", skiprows=7, nrows=15)

            sectors = df_returns.columns.tolist()
            returns = df_returns.iloc[0].values / 100
            stdevs = df_volatility.iloc[0].values / 100
            corr_matrix = df_corr.iloc[:, 1:].values
            cov_matrix = np.outer(stdevs, stdevs) * corr_matrix

            # Run simulations
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(sectors))
                weights /= np.sum(weights)
                weights_record.append(weights)
                portfolio_return = np.sum(weights * returns)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_stddev
                results[0,i] = portfolio_return
                results[1,i] = portfolio_stddev
                results[2,i] = sharpe_ratio

            results_df = pd.DataFrame(results.T, columns=["Return", "Risk", "Sharpe"])
            weights_df = pd.DataFrame(weights_record, columns=sectors)
            final_df = pd.concat([results_df, weights_df], axis=1)

            st.subheader("Efficient Frontier")
            fig, ax = plt.subplots(figsize=(10,6))
            scatter = ax.scatter(final_df["Risk"], final_df["Return"], c=final_df["Sharpe"], cmap="viridis", alpha=0.6)
            ax.set_xlabel("Risk (Std. Deviation)")
            ax.set_ylabel("Expected Return")
            ax.set_title("Simulated Portfolio Optimization")
            fig.colorbar(scatter, label="Sharpe Ratio")
            st.pyplot(fig)

            st.subheader("Optimization Data Preview")
            st.dataframe(final_df.head(10).style.format({"Return": "{:.2%}", "Risk": "{:.2%}", "Sharpe": "{:.2f}"}))

            # Download link
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False, sheet_name="CAPM Optimization")
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="capm_output.xlsx">📥 Download Results as Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
