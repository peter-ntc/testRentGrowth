
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# File upload
st.title("Efficient Frontier Optimizer")
uploaded_file = st.file_uploader("Upload sector_input.xlsx", type=["xlsx"])

if uploaded_file is not None:
    # Load Excel using openpyxl
    wb = load_workbook(uploaded_file, data_only=True)
    ws = wb.active

    def read_row(row_num, start_col, end_col):
        return [ws.cell(row=row_num, column=col).value for col in range(start_col, end_col + 1)]

    # Read inputs
    sectors = read_row(3, 2, 17)
    expected_returns = np.array(read_row(4, 2, 17), dtype=float)
    volatility = np.array(read_row(5, 2, 17), dtype=float)
    cor_matrix = np.array([read_row(r, 2, 17) for r in range(9, 25)], dtype=float)
    min_weights = np.array(read_row(110, 2, 17), dtype=float)
    max_weights = np.array(read_row(111, 2, 17), dtype=float)
    risk_free_rate = ws["B113"].value or 0.0

    # Calculate covariance matrix
    D = np.diag(volatility)
    cov_matrix = D @ cor_matrix @ D

    # Portfolio simulation
    num_portfolios = 10000
    results = np.zeros((4 + len(sectors), num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.uniform(size=len(sectors))
        weights = weights / np.sum(weights)
        if np.any(weights < min_weights) or np.any(weights > max_weights):
            continue
        port_return = np.dot(weights, expected_returns)
        port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_stddev if port_stddev != 0 else 0
        results[0,i] = port_return
        results[1,i] = port_stddev
        results[2,i] = sharpe_ratio
        results[3:,i] = weights

    results_df = pd.DataFrame(results.T, columns=["Return", "Risk", "Sharpe"] + sectors)
    max_sharpe_port = results_df.iloc[results_df["Sharpe"].idxmax()]
    min_risk_port = results_df.iloc[results_df["Risk"].idxmin()]

    st.subheader("Max Sharpe Portfolio Allocation")
    st.dataframe(max_sharpe_port[3:].apply(lambda x: f"{x:.2%}"))

    st.subheader("Min Risk Portfolio Allocation")
    st.dataframe(min_risk_port[3:].apply(lambda x: f"{x:.2%}"))

    # Plot Efficient Frontier
    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(results_df["Risk"], results_df["Return"], c=results_df["Sharpe"], cmap='viridis', alpha=0.7)
    ax1.scatter(max_sharpe_port["Risk"], max_sharpe_port["Return"], c='red', marker='*', s=200, label='Max Sharpe')
    ax1.scatter(min_risk_port["Risk"], min_risk_port["Return"], c='blue', marker='X', s=100, label='Min Risk')
    ax1.set_xlabel("Volatility (Risk)")
    ax1.set_ylabel("Expected Return")
    ax1.set_title("Efficient Frontier with Monte Carlo Simulation")
    ax1.legend()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))
    st.pyplot(fig1)
