
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set page config
st.set_page_config(page_title="Efficient Frontier", layout="wide")
st.title("Efficient Frontier Generator")

uploaded_file = st.file_uploader("Upload the Excel file (.xlsx) with a sheet named 'EffFrontInputs'", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xls, sheet_name="EffFrontInputs", header=None)

        n = int(df.iloc[1, 1])
        if n > 100:
            st.error("The number of assets (n) exceeds the maximum allowed value of 100.")
            st.stop()

        expected_returns = df.iloc[3, 1:1+n].values.astype(float)
        volatilities = df.iloc[4, 1:1+n].values.astype(float)
        covariance_matrix = df.iloc[8:8+n, 1:1+n].values.astype(float)
        min_weights = df.iloc[109, 1:1+n].values.astype(float)
        max_weights = df.iloc[110, 1:1+n].values.astype(float)
        risk_free_rate = float(df.iloc[112, 1])
        sector_names = df.iloc[2, 1:1+n].values.astype(str).tolist()

        def portfolio_performance(weights):
            returns = np.dot(weights, expected_returns)
            std = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            return returns, std

        def minimize_volatility(target_return):
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
            )
            bounds = tuple((min_weights[i], max_weights[i]) for i in range(n))
            result = minimize(
                lambda x: portfolio_performance(x)[1],
                x0=np.repeat(1/n, n),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            return result

        target_returns = np.linspace(min(expected_returns), max(expected_returns), 100)
        frontier_volatilities = []
        weights_list = []

        for r in target_returns:
            result = minimize_volatility(r)
            if result.success:
                frontier_volatilities.append(result.fun)
                weights_list.append(result.x)
            else:
                frontier_volatilities.append(np.nan)
                weights_list.append([np.nan] * n)

        sharpe_ratios = (target_returns - risk_free_rate) / np.array(frontier_volatilities)

        frontier_df = pd.DataFrame({
            "Return": target_returns,
            "Volatility": frontier_volatilities,
            "Sharpe Ratio": sharpe_ratios
        })
        for i in range(n):
            frontier_df[sector_names[i]] = [w[i] for w in weights_list]

        frontier_df = frontier_df.sort_values("Volatility")
        filtered_df = frontier_df.loc[frontier_df["Return"].cummax() == frontier_df["Return"]].reset_index(drop=True)

        max_sharpe_idx = filtered_df["Sharpe Ratio"].idxmax()
        optimal_weights = filtered_df.loc[max_sharpe_idx, sector_names]
        optimal_return = filtered_df.loc[max_sharpe_idx, "Return"]
        optimal_volatility = filtered_df.loc[max_sharpe_idx, "Volatility"]
        optimal_sharpe = filtered_df.loc[max_sharpe_idx, "Sharpe Ratio"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filtered_df["Volatility"], filtered_df["Return"], label='Efficient Frontier', color='blue')
        ax.scatter(optimal_volatility, optimal_return, c='red', marker='*', s=200, label='Max Sharpe Ratio')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier (Upper Segment)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Maximum Sharpe Ratio Portfolio")
        for i in range(n):
            st.write(f"{sector_names[i]}: {optimal_weights[sector_names[i]]:.2%}")
        st.write(f"Expected Return: {optimal_return:.2%}")
        st.write(f"Volatility: {optimal_volatility:.2%}")
        st.write(f"Sharpe Ratio: {optimal_sharpe:.2f}")

        st.subheader("Sector Mix at Selected Volatility Levels")

        sampled_vols = np.linspace(filtered_df["Volatility"].min(), filtered_df["Volatility"].max(), 20)
        if optimal_volatility not in sampled_vols:
            sampled_vols = np.sort(np.append(sampled_vols, optimal_volatility))

        sector_mix_table = pd.DataFrame()
        for v in sampled_vols:
            closest_idx = (filtered_df["Volatility"] - v).abs().idxmin()
            row_data = {
                "Volatility": filtered_df.loc[closest_idx, "Volatility"],
                "Expected Return": filtered_df.loc[closest_idx, "Return"],
                "Sharpe Ratio": filtered_df.loc[closest_idx, "Sharpe Ratio"],
            }
            row_data.update({
                sector_names[i]: filtered_df.loc[closest_idx, sector_names[i]]
                for i in range(n)
            })
            sector_mix_table = pd.concat([sector_mix_table, pd.DataFrame([row_data])], ignore_index=True)

        max_sharpe_row_idx = sector_mix_table["Sharpe Ratio"].idxmax()
        styled = sector_mix_table.style.format({col: "{:.2%}" for col in sector_names})
        styled = styled.format({"Volatility": "{:.2%}", "Expected Return": "{:.2%}", "Sharpe Ratio": "{:.2f}"})
        styled = styled.apply(lambda x: ['font-weight: bold' if x.name == max_sharpe_row_idx else '' for _ in x], axis=1)

        st.dataframe(styled)

    except Exception as e:
        st.error(f"Error processing file: {e}")
