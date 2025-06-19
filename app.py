
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="CAPM Optimizer", layout="wide")

def run_optimizer():
    try:
        df = pd.read_excel("capm_input.xlsx", header=None)

        sectors = df.iloc[2, 1:15].tolist()
        sectors = [s.strip() for s in sectors]
        returns = df.iloc[3, 1:15].astype(float).values
        volatilities = df.iloc[4, 1:15].astype(float).values
        corr_matrix = df.iloc[8:22, 1:15].astype(float).values
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        num_assets = len(sectors)

        def portfolio_perf(weights):
            port_return = np.dot(weights, returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return port_return, port_std

        def objective(weights):
            _, std = portfolio_perf(weights)
            return std

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]

        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            opt_weights = result.x
            exp_return, exp_std = portfolio_perf(opt_weights)

            st.subheader("Efficient Frontier")
            plt.figure(figsize=(8, 5))
            plt.bar(sectors, opt_weights)
            plt.xticks(rotation=45, ha="right")
            plt.title("Optimal Portfolio Weights")
            plt.tight_layout()
            st.pyplot(plt.gcf())

            st.subheader("Portfolio Statistics")
            st.write("**Expected Return:** {:.2%}".format(exp_return))
            st.write("**Risk (Standard Deviation):** {:.2%}".format(exp_std))

            output_df = pd.DataFrame({
                "Sector": sectors,
                "Weight": opt_weights
            })
            st.download_button("Download Results (CSV)", data=output_df.to_csv(index=False), file_name="capm_output.csv")
        else:
            st.error("Optimization failed.")

    except Exception as e:
        st.error(f"Failed to load or process 'capm_input.xlsx'. Error: {e}")

st.image("townsendAI_logo_1.png", width=100)
st.title("📈 CAPM Optimization")
st.subheader("Efficient Frontier")

if st.button("Run Optimization"):
    run_optimizer()
