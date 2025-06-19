import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import BytesIO

def run_capm_optimizer():
    df_returns = pd.read_excel("capm input.xlsx", sheet_name=0, usecols="B:O", nrows=2)
    df_returns.index = ["Expected Return", "Volatility"]
    sectors = pd.read_excel("capm input.xlsx", sheet_name=0, usecols="B:O", nrows=1, header=None).values.flatten()
    df_returns.columns = sectors

    df_corr = pd.read_excel("capm input.xlsx", sheet_name=0, skiprows=7, usecols="B:O", nrows=14, header=None)
    df_corr.columns = sectors
    df_corr.index = sectors

    mean_returns = df_returns.loc["Expected Return"]
    volatilities = df_returns.loc["Volatility"]
    cov_matrix = np.outer(volatilities, volatilities) * df_corr.to_numpy()
    cov_df = pd.DataFrame(cov_matrix, index=sectors, columns=sectors)

    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.dot(weights, mean_returns)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std_dev

    def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
        p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_return - risk_free_rate) / p_std

    num_assets = len(sectors)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = num_assets * [1. / num_assets,]

    optimized = minimize(negative_sharpe, initial_weights,
                         args=(mean_returns, cov_matrix), method='SLSQP',
                         bounds=bounds, constraints=constraints)

    opt_weights = optimized.x
    opt_return, opt_std = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = (portfolio_return) / portfolio_std

    plt.figure(figsize=(10,6))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(opt_std, opt_return, marker='*', color='r', s=100, label='Max Sharpe Ratio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.legend()

    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    plt.close()

    output_df = pd.DataFrame({"Sector": sectors, "Weight": opt_weights})
    output_df["Weight"] = output_df["Weight"].map(lambda x: f"{x:.2%}")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name='Optimal Weights')
        writer.sheets['Optimal Weights'].set_column('A:B', 20)
    excel_buffer.seek(0)

    return image_buffer, excel_buffer