
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO

def run_capm_optimizer(expected_returns, volatilities, corr_matrix):
    try:
        cov_matrix = corr_matrix.values * np.outer(volatilities, volatilities)

        num_assets = len(expected_returns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # weights sum to 1
        )

        def portfolio_return(weights):
            return np.dot(weights, expected_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def negative_sharpe_ratio(weights):
            return -portfolio_return(weights) / portfolio_volatility(weights)

        init_guess = np.repeat(1 / num_assets, num_assets)
        result = minimize(negative_sharpe_ratio, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        expected_portfolio_return = portfolio_return(optimal_weights)
        expected_portfolio_volatility = portfolio_volatility(optimal_weights)

        # Efficient Frontier
        target_returns = np.linspace(min(expected_returns), max(expected_returns), 50)
        frontier_vols = []

        for r in target_returns:
            constraints_rf = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x: np.dot(x, expected_returns) - r},
            )
            result_rf = minimize(portfolio_volatility, init_guess, method="SLSQP", bounds=bounds, constraints=constraints_rf)
            frontier_vols.append(portfolio_volatility(result_rf.x))

        fig1, ax1 = plt.subplots()
        ax1.plot(frontier_vols, target_returns, "b--", label="Efficient Frontier")
        ax1.set_xlabel("Volatility (Std. Dev.)")
        ax1.set_ylabel("Expected Return")
        ax1.set_title("Efficient Frontier")
        ax1.legend()

        frontier_img = BytesIO()
        fig1.savefig(frontier_img, format="png", bbox_inches="tight")
        frontier_img.seek(0)
        plt.close(fig1)

        # Weights Chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(range(num_assets), optimal_weights, tick_label=expected_returns.index)
        ax2.set_ylabel("Weight")
        ax2.set_title("Optimal Portfolio Weights")
        ax2.set_xticklabels(expected_returns.index, rotation=45, ha='right')

        weights_img = BytesIO()
        fig2.savefig(weights_img, format="png", bbox_inches="tight")
        weights_img.seek(0)
        plt.close(fig2)

        # Excel output
        output_df = pd.DataFrame({
            "Sector": expected_returns.index,
            "Weight": optimal_weights,
            "Expected Return": expected_returns,
            "Volatility": volatilities
        })

        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            output_df.to_excel(writer, sheet_name="Portfolio", index=False)
            summary = pd.DataFrame({
                "Expected Portfolio Return": [expected_portfolio_return],
                "Portfolio Volatility": [expected_portfolio_volatility]
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)
        excel_file.seek(0)

        return frontier_img, weights_img, excel_file

    except Exception as e:
        raise RuntimeError(f"Optimization failed: {str(e)}")
