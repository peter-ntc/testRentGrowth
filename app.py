import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize

# UI: Title and file upload
st.title("Portfolio Optimizer with Efficient Frontier")
uploaded_file = st.file_uploader("Upload your Excel file (e.g., sector_input.xlsx)", type=["xlsx"])

if uploaded_file:
    sheet = pd.read_excel(uploaded_file, header=None)

    # Extract inputs
    sectors = sheet.loc[2, 1:16].values
    expected_returns = sheet.loc[3, 1:16].astype(float).values
    volatility = sheet.loc[4, 1:16].astype(float).values
    cor_matrix = sheet.loc[8:23, 1:16].astype(float).values
    min_weights = sheet.loc[109, 1:16].astype(float).values
    max_weights = sheet.loc[110, 1:16].astype(float).values
    risk_free_rate = float(sheet.loc[112, 1])


    # Covariance matrix
    i_lower = np.tril_indices_from(cor_matrix, -1)
    cor_matrix[i_lower[::-1]] = cor_matrix[i_lower]
    D = np.diag(volatility)
    cov_matrix = D @ cor_matrix @ D

    # Efficient Frontier
    num_assets = len(expected_returns)
    num_points = 50
    target_returns = np.linspace(min(expected_returns), max(expected_returns), num_points)
    frontier_risks = []
    frontier_weights = []

    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ expected_returns - target}
        ]
        bounds = tuple(zip(min_weights, max_weights))
        init_guess = np.repeat(1 / num_assets, num_assets)

        result = minimize(lambda w: w.T @ cov_matrix @ w, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            frontier_risks.append(np.sqrt(result.fun))
            frontier_weights.append(result.x)
        else:
            frontier_risks.append(np.nan)
            frontier_weights.append([np.nan] * num_assets)

    # Sharpe Ratios
    target_returns = np.array(target_returns)
    frontier_risks = np.array(frontier_risks)
    sharpe_ratios = (target_returns - risk_free_rate) / frontier_risks
    max_idx = np.nanargmax(sharpe_ratios)

    df_output = pd.DataFrame(frontier_weights, columns=sectors)
    df_output.insert(0, "Expected Return", target_returns)
    df_output.insert(1, "Portfolio Risk", frontier_risks)
    df_output["Sharpe Ratio"] = sharpe_ratios

    st.subheader("Efficient Portfolios Table")
    st.dataframe(df_output.style.format("{:.2%}"))

    # Monte Carlo Simulation
    sim_weights = np.random.dirichlet(np.ones(num_assets), size=10000)
    sim_returns = sim_weights @ expected_returns
    sim_risks = np.sqrt(np.einsum('ij,jk,ik->i', sim_weights, cov_matrix, sim_weights))
    sim_sharpes = (sim_returns - risk_free_rate) / sim_risks

    # Plot: Efficient Frontier with Monte Carlo
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sc = ax1.scatter(sim_risks, sim_returns, c=sim_sharpes, cmap='coolwarm', alpha=0.3)
    ax1.plot(frontier_risks, target_returns, color='black', linewidth=2, label='Efficient Frontier')
    ax1.scatter(frontier_risks[max_idx], target_returns[max_idx], color='gold', s=100, edgecolor='black', label='Max Sharpe')
    ax1.set_title("Efficient Frontier with Monte Carlo Overlay")
    ax1.set_xlabel("Portfolio Risk")
    ax1.set_ylabel("Expected Return")
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

    # Plot: Sector Weight Allocation
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 3]})
    ax2a.plot(frontier_risks, target_returns, marker='o', color='black', label="Efficient Frontier")
    ax2a.set_title('Efficient Frontier with Sector Mix')
    ax2a.set_xlabel('Portfolio Risk')
    ax2a.set_ylabel('Expected Return')
    ax2a.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2a.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2a.grid(True)
    ax2a.legend()

    weights_array = np.array(frontier_weights)
    ax2b.stackplot(target_returns, weights_array.T, labels=sectors)
    ax2b.set_title("Sector Weight Allocation Across the Efficient Frontier")
    ax2b.set_xlabel("Expected Return")
    ax2b.set_ylabel("Weight")
    ax2b.set_ylim(0, 1)
    ax2b.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2b.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2b.grid(True)
    ax2b.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig2.tight_layout()
    st.pyplot(fig2)
