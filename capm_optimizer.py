
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os

def run_capm_optimizer(input_file="capm input.xlsx", output_file="capm_output.xlsx"):
    frontier_path = os.path.abspath("efficient_frontier.png")
    weights_path = os.path.abspath("weights_stackplot.png")
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import matplotlib.ticker as mtick
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import numbers
    
    # Load input Excel file
    input_path = "capm input.xlsx"
    sheet = pd.read_excel(input_path, header=None)
    
    # Extract inputs
    sectors = sheet.loc[2, 1:15].values
    expected_returns = sheet.loc[3, 1:15].astype(float).values
    volatility = sheet.loc[4, 1:15].astype(float).values
    cor_matrix = sheet.loc[8:21, 1:14].astype(float).values
    min_weights = sheet.loc[24, 1:15].astype(float).values
    max_weights = sheet.loc[25, 1:15].astype(float).values
    risk_free_rate = float(sheet.loc[35, 1])  # Cell B36
    
    # Build covariance matrix
    i_lower = np.tril_indices_from(cor_matrix, -1)
    cor_matrix[i_lower[::-1]] = cor_matrix[i_lower]
    D = np.diag(volatility)
    cov_matrix = D @ cor_matrix @ D
    
    # Compute Efficient Frontier
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
    
    # Sharpe Ratios and Max Sharpe Portfolio
    target_returns = np.array(target_returns)
    frontier_risks = np.array(frontier_risks)
    sharpe_ratios = (target_returns - risk_free_rate) / frontier_risks
    
    df_output = pd.DataFrame(frontier_weights, columns=sectors)
    df_output.insert(0, "Expected Return", target_returns)
    df_output.insert(1, "Portfolio Risk", frontier_risks)
    df_output["Sharpe Ratio"] = sharpe_ratios
    max_idx = np.nanargmax(sharpe_ratios)
    
    # Monte Carlo Simulation
    sim_weights = np.random.dirichlet(np.ones(num_assets), size=10000)
    sim_returns = sim_weights @ expected_returns
    sim_risks = np.sqrt(np.einsum('ij,jk,ik->i', sim_weights, cov_matrix, sim_weights))
    sim_sharpes = (sim_returns - risk_free_rate) / sim_risks
    
    # Plot 1: Monte Carlo + Frontier
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sc = ax1.scatter(sim_risks, sim_returns, c=sim_sharpes, cmap='coolwarm', alpha=0.3, label='Simulated Portfolios')
    ax1.plot(frontier_risks, target_returns, color='black', linewidth=2, label='Efficient Frontier')
    ax1.scatter(frontier_risks[max_idx], target_returns[max_idx], color='gold', s=100, edgecolor='black', label='Max Sharpe')
    ax1.set_title("Efficient Frontier with Monte Carlo Overlay")
    ax1.set_xlabel("Portfolio Risk")
    ax1.set_ylabel("Expected Return")
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.legend()
    plt.colorbar(sc, label='Sharpe Ratio')
    fig1.tight_layout()
    monte_carlo_path = "efficient_frontier_monte_carlo.png"
    fig1.savefig(monte_carlo_path)
    plt.show()
    
    # Plot 2: Sector Allocation Stackplot
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
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
    allocation_path = "efficient_frontier_allocation.png"
    fig2.savefig(allocation_path)
    plt.show()
    
    # Write outputs to Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "output"
    
    for r_idx, row in enumerate(dataframe_to_rows(df_output, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            if r_idx > 1:
                cell.number_format = '0.00%'
    
    img1 = XLImage(monte_carlo_path)
    img1.width *= 0.5
    img1.height *= 0.5
    ws.add_image(img1, "R2")
    
    img2 = XLImage(allocation_path)
    img2.width *= 0.5
    img2.height *= 0.5
    ws.add_image(img2, "R35")
    
    wb.save("capm_output.xlsx")
    return output_file, frontier_path, weights_path