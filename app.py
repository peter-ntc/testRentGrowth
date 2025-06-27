import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).parent

if "page" not in st.session_state:
    st.session_state.page = "home"
if "scenario" not in st.session_state:
    st.session_state.scenario = None

def go_home():
    st.session_state.page = "home"
    st.session_state.scenario = None

def set_page(option):
    st.session_state.page = f"option{option}"
    st.session_state.scenario = None

def set_scenario(name):
    st.session_state.scenario = name

def render_scenario_table(filename, key_prefix=""):
    try:
        df = pd.read_excel(BASE_DIR / filename, header=None, usecols="C,D,K,L,N,O,P,Q,R,S,V,W,AE,AF", skiprows=2, nrows=19)
        df.columns = [
            "Sector", "Entry Cap Rate", "Exit Cap Rate Year 3", "Exit Cap Rate Year 6",
            "Rent Growth Year 1", "Rent Growth Year 2", "Rent Growth Year 3",
            "Rent Growth Year 4", "Rent Growth Year 5", "Rent Growth Year 6",
            "3 Yr Unlevered Property Returns", "6 Yr Unlevered Property Returns",
            "3 Yr Levered Fund Net Returns", "6 Yr Levered Fund Net Returns"
        ]
        for col in df.columns[1:]:
            df[col] = df[col].apply(lambda x: f"{x:.2%}")
        st.markdown("### Sector Summary Table")
        st.dataframe(df, use_container_width=True, height=800, key=f"{key_prefix}_table")
    except Exception as e:
        st.error(f"Error loading table: {e}")

def safe_load_df(filename, label):
    try:
        df = pd.read_excel(BASE_DIR / filename, header=None, usecols="D:I", skiprows=52, nrows=3)
        df.columns = ["2025", "2026", "2027", "2028", "2029", "2030"]
        df.index = ["GDP", "Inflation", "10 YR"]
        return df.astype(float)
    except Exception as e:
        st.error(f"Failed to load {label}: {e}")
        return None

def plot_chart(data, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in data.index:
        ax.plot(data.columns, data.loc[row] * 100, label=row)
    ax.set_title(title)
    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

def render_forecasting_modeling():
    if st.session_state.scenario is None:
        st.title("Forecasting & Modeling")
        st.markdown("### Forecasting and Modeling supported for 3 economic scenarios.")

        row1_left, row1_right = st.columns([1, 3])
        with row1_left:
            st.button("Consensus Economic Outlook", on_click=set_scenario, args=("consensus",), key="btn_consensus", use_container_width=True)
        with row1_right:
            df_base = safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook")
            if df_base is not None:
                plot_chart(df_base, "Consensus Economic Outlook")

        row2_left, row2_right = st.columns([1, 3])
        with row2_left:
            st.button("Higher Growth & Inflation", on_click=set_scenario, args=("high",), key="btn_high", use_container_width=True)
        with row2_right:
            df_high = safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation")
            if df_high is not None:
                plot_chart(df_high, "Higher Growth & Inflation")

        row3_left, row3_right = st.columns([1, 3])
        with row3_left:
            st.button("Lower Growth & Inflation", on_click=set_scenario, args=("low",), key="btn_low", use_container_width=True)
        with row3_right:
            df_low = safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation")
            if df_low is not None:
                plot_chart(df_low, "Lower Growth & Inflation")

        row4_left, row4_right = st.columns([1, 3])
        with row4_left:
            st.button("Compare ALL 3 scenarios", on_click=set_scenario, args=("compare",), key="btn_compare_main", use_container_width=True)
        with row4_right:
            try:
                base = safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook")
                high = safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation")
                low = safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation")

                def avg(df, label):
                    return df.loc[label].mean() if df is not None else 0

                metrics = ["GDP", "Inflation", "10 YR"]
                consensus = [avg(base, m) * 100 for m in metrics]
                high_growth = [avg(high, m) * 100 for m in metrics]
                low_growth = [avg(low, m) * 100 for m in metrics]

                x = np.arange(len(metrics))
                width = 0.25

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.bar(x - width, consensus, width, label="Consensus")
                ax.bar(x, high_growth, width, label="High")
                ax.bar(x + width, low_growth, width, label="Low")

                ax.set_ylabel("Average (%)")
                ax.set_title("Comparison of Average Metrics (2025–2030)")
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Unable to render comparison chart: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_overview")

    else:
        label = st.session_state.scenario
        if label == "consensus":
            st.subheader("Consensus Economic Outlook")
            df = safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook")
            if df is not None:
                plot_chart(df, "Consensus Economic Outlook")
            render_scenario_table("BaseScenario.xlsx", key_prefix="consensus")
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_consensus")
        elif label == "high":
            st.subheader("Higher Growth & Inflation")
            df = safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation")
            if df is not None:
                plot_chart(df, "Higher Growth & Inflation")
            render_scenario_table("HighScenario.xlsx", key_prefix="high")
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_high")
        elif label == "low":
            st.subheader("Lower Growth & Inflation")
            df = safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation")
            if df is not None:
                plot_chart(df, "Lower Growth & Inflation")
            render_scenario_table("LowScenario.xlsx", key_prefix="low")
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_low")
        
        elif label == "compare":
            st.subheader("Compare ALL 3 scenarios")

            def load_returns(file):
                try:
                    df = pd.read_excel(BASE_DIR / file, header=None, usecols="C,V,W,AE,AF", skiprows=2, nrows=19)
                    df.columns = ["Sector", "Unlev_3yr", "Unlev_6yr", "Lev_3yr", "Lev_6yr"]
                    return df
                except:
                    return pd.DataFrame()

            base_df = load_returns("BaseScenario.xlsx")
            high_df = load_returns("HighScenario.xlsx")
            low_df = load_returns("LowScenario.xlsx")

            sectors = list(base_df["Sector"]) if not base_df.empty else []
            selected = st.multiselect("Select up to 5 sectors to compare:", sectors, max_selections=5)

            if selected:
                def extract(df, col):
                    return df[df["Sector"].isin(selected)].set_index("Sector")[col]

                b_unlev = extract(base_df, "Unlev_6yr")
                h_unlev = extract(high_df, "Unlev_6yr")
                l_unlev = extract(low_df, "Unlev_6yr")

                b_lev = extract(base_df, "Lev_6yr")
                h_lev = extract(high_df, "Lev_6yr")
                l_lev = extract(low_df, "Lev_6yr")

                x = np.arange(len(selected))
                width = 0.25

                # Chart 1: Unlevered Returns
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.bar(x - width, b_unlev * 100, width, label="Base")
                ax1.bar(x, h_unlev * 100, width, label="High")
                ax1.bar(x + width, l_unlev * 100, width, label="Low")
                ax1.set_xticks(x)
                ax1.set_xticklabels(selected, rotation=45, ha='right')
                ax1.set_ylabel("Unlevered Returns (%)")
                ax1.set_title("Unlevered Property-Level Returns (6 Yr)")
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
                ax1.legend()
                fig1.tight_layout()
                st.pyplot(fig1)

                # Chart 2: Levered Returns
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.bar(x - width, b_lev * 100, width, label="Base")
                ax2.bar(x, h_lev * 100, width, label="High")
                ax2.bar(x + width, l_lev * 100, width, label="Low")
                ax2.set_xticks(x)
                ax2.set_xticklabels(selected, rotation=45, ha='right')
                ax2.set_ylabel("Levered Returns (%)")
                ax2.set_title("Levered Fund Net Returns (6 Yr)")
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
                ax2.legend()
                fig2.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("Please select up to 5 sectors to view comparison.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_compare")



def render_option(option_num):
    if option_num == "1":
        render_forecasting_modeling()

    elif option_num == "2":
        if st.session_state.scenario == "capm":
            render_capm()
        elif st.session_state.scenario == "model_portfolio":
            render_model_portfolio()
        else:
            st.title("Optimizer")
            st.subheader("Choose a method to begin optimization:")
            col1, col2 = st.columns(2)
            with col1:
                st.button("CAPM", on_click=set_scenario, args=("capm",), key="btn_opt_capm", use_container_width=True)
            with col2:
                st.button("Model Portfolio", on_click=set_scenario, args=("model_portfolio",), key="btn_opt_model", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_optimizer")

    elif option_num == "3":
        render_fund_pipeline()

    elif option_num == "4":
        render_smart_benchmarks()

    elif option_num == "5":
        st.title("Secondaries Marketplace")
        st.subheader("🚧 Under Construction 🚧")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_option5")

    elif option_num == "6":
        render_market_research()



def landing_page():
    logo_path = BASE_DIR / "townsendAI_logo_1.png"
    if logo_path.exists():
        logo = Image.open(logo_path)
        colA, colB, colC = st.columns([1, 2, 1])
        with colB:
            st.image(logo, width=250)

    st.title("TownsendAI")
    st.write("Welcome to the MVP. Please select an option:")

    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 100px;
        font-size: 18px;
        border-radius: 10px;
        white-space: normal;
    }
    </style>
    """, unsafe_allow_html=True)

    option_labels = [
        "Forecasting & Modeling",
        "Optimizer",
        "Fund & Deal Pipeline",
        "Smart Benchmarks",
        "Secondaries Marketplace",
        "Market Research"
    ]

    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3], start=0):
        with col:
            st.button(option_labels[i], on_click=set_page, args=(i+1,), key=f"btn_{i}")

    col4, col5, col6 = st.columns(3)
    for i, col in enumerate([col4, col5, col6], start=3):
        with col:
            st.button(option_labels[i], on_click=set_page, args=(i+1,), key=f"btn_{i}")



def render_capm():

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from scipy.optimize import minimize

    # UI: Title and file upload
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("Portfolio Optimizer with Efficient Frontier")
    uploaded_file = st.file_uploader("Upload your Excel file (e.g., capm_input.xlsx)", type=["xlsx"])

    if uploaded_file:
        sheet = pd.read_excel(uploaded_file, header=None)

        # Extract inputs
        sectors = sheet.loc[2, 1:15].values
        expected_returns = sheet.loc[3, 1:15].astype(float).values
        volatility = sheet.loc[4, 1:15].astype(float).values
        cor_matrix = sheet.loc[8:21, 1:14].astype(float).values
        min_weights = sheet.loc[24, 1:15].astype(float).values
        max_weights = sheet.loc[25, 1:15].astype(float).values
        risk_free_rate = float(sheet.loc[35, 1])

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

def render_model_portfolio():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Set page config

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



def render_fund_pipeline():
    st.title("🏦 Fund & Deal Pipeline Analytics")
    st.markdown("Explore and analyze your investment pipeline with advanced filtering and visualization tools.")
    
    # File upload section with sample data option
    uploaded_file = st.file_uploader("Upload Pipeline Excel File", type=["xlsx"], 
                                   help="Upload an Excel file with 'Pipeline' sheet containing your deal data")
    
    if not uploaded_file:
        if st.button("Use Sample Data", help="Try with sample pipeline data"):
            sample_data = {
                "Property Type": ["Residential", "Office", "Retail", "Industrial", "Mixed Use"],
                "Entity Invest.": ["Yes", "No", "Yes", "No", "Yes"],
                "Strategic": ["Yes", "No", "Yes", "Yes", "No"],
                "Synd.": ["No", "Yes", "No", "Yes", "No"],
                "Co-Invest Equity": [500000, 1200000, 750000, 2000000, 350000],
                "Gross IRR": [0.15, 0.12, 0.18, 0.10, 0.22],
                "Gross EM": [1.8, 1.5, 2.1, 1.3, 2.4],
                "Fund Name": ["Alpha Fund", "Beta Fund", "Gamma Fund", "Delta Fund", "Epsilon Fund"],
                "Vintage Year": [2020, 2021, 2022, 2021, 2023]
            }
            df = pd.DataFrame(sample_data)
            st.session_state.pipeline_df = df
            st.success("Sample data loaded successfully!")
        return
    
    try:
        # Load and prepare data
        if 'pipeline_df' not in st.session_state or st.button("Reload Data"):
            df = pd.read_excel(uploaded_file, sheet_name="Pipeline")
            
            # Convert numeric columns and handle missing values
            numeric_cols = ['Gross IRR', 'Gross EM', 'Co-Invest Equity']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            st.session_state.pipeline_df = df
            st.success("Data loaded successfully!")
        
        df = st.session_state.pipeline_df
        
        # Quick stats overview
        with st.expander("📊 Pipeline Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Deals", len(df))
            col2.metric("Avg IRR", f"{df['Gross IRR'].mean():.1%}")
            col3.metric("Avg Equity", f"${df['Co-Invest Equity'].mean():,.0f}")
        
        # Filter controls in a sidebar
        with st.sidebar:
            st.subheader("🔍 Filter Options")
            
            # Text search filters
            prop_type = st.text_input("Property Type Filter", "")
            fund_name = st.text_input("Fund Name Filter", "")
            
            # Boolean filters
            bool_filters = {
                "Entity Invest.": ["Any", "Yes", "No"],
                "Strategic": ["Any", "Yes", "No"],
                "Synd.": ["Any", "Yes", "No"]
            }
            
            bool_selections = {}
            for col, options in bool_filters.items():
                if col in df.columns:
                    bool_selections[col] = st.radio(col, options, horizontal=True)
            
            # Numeric range filters
            st.markdown("---")
            st.subheader("Financial Metrics")
            
            if 'Co-Invest Equity' in df.columns:
                coinv_min, coinv_max = int(df["Co-Invest Equity"].min()), int(df["Co-Invest Equity"].max())
                coinv_range = st.slider("Co-Invest Equity ($)", 
                                       coinv_min, coinv_max, 
                                       (coinv_min, coinv_max),
                                       help="Filter by investment amount range")
            
            if 'Gross IRR' in df.columns:
                irr_range = st.slider("Gross IRR (%) Range", 
                                     float(df["Gross IRR"].min()), 
                                     float(df["Gross IRR"].max()), 
                                     (float(df["Gross IRR"].min()), 
                                      float(df["Gross IRR"].max())),
                                     format="%.2f")
            
            if 'Gross EM' in df.columns:
                em_range = st.slider("Gross EM (x) Range", 
                                   float(df["Gross EM"].min()), 
                                   float(df["Gross EM"].max()), 
                                   (float(df["Gross EM"].min()), 
                                    float(df["Gross EM"].max())),
                                   format="%.2f")
        
        # Apply filters
        filtered_df = df.copy()
        
        # Text filters
        if prop_type:
            filtered_df = filtered_df[filtered_df["Property Type"].str.contains(prop_type, case=False, na=False)]
        if fund_name:
            filtered_df = filtered_df[filtered_df["Fund Name"].str.contains(fund_name, case=False, na=False)]
        
        # Boolean filters
        for col, selection in bool_selections.items():
            if selection != "Any":
                val = selection == "Yes"
                filtered_df = filtered_df[filtered_df[col].fillna("").str.lower() == str(val).lower()]
        
        # Numeric filters
        if 'Co-Invest Equity' in df.columns:
            filtered_df = filtered_df[filtered_df["Co-Invest Equity"].between(coinv_range[0], coinv_range[1])]
        if 'Gross IRR' in df.columns:
            filtered_df = filtered_df[filtered_df["Gross IRR"].between(irr_range[0], irr_range[1])]
        if 'Gross EM' in df.columns:
            filtered_df = filtered_df[filtered_df["Gross EM"].between(em_range[0], em_range[1])]
        
        # Results display
        st.subheader(f"📋 Filtered Results ({len(filtered_df)} deals)")
        
        if not filtered_df.empty:
            # Tabbed interface for different views
            tab1, tab2, tab3 = st.tabs(["Data Table", "Visualizations", "Export"])
            
            with tab1:
                # Enhanced dataframe display
                gb = st.grid(
                    [3,1], 
                    vertical_align="bottom",
                    gap="small"
                )
                gb.dataframe(
                    filtered_df.style.format({
                        'Gross IRR': '{:.1%}',
                        'Gross EM': '{:.1f}x',
                        'Co-Invest Equity': '${:,.0f}'
                    }),
                    use_container_width=True,
                    height=600
                )
                
                gb.download_button(
                    "Download Filtered Data",
                    filtered_df.to_csv(index=False).encode('utf-8'),
                    "filtered_pipeline.csv",
                    "text/csv"
                )
            
            with tab2:
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Property Type' in filtered_df.columns:
                        st.subheader("By Property Type")
                        fig1 = px.pie(
                            filtered_df, 
                            names='Property Type', 
                            values='Co-Invest Equity',
                            hole=0.4
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    if 'Gross IRR' in filtered_df.columns:
                        st.subheader("IRR Distribution")
                        fig2 = px.histogram(
                            filtered_df, 
                            x='Gross IRR',
                            nbins=10,
                            color='Property Type' if 'Property Type' in filtered_df.columns else None
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                
                if 'Vintage Year' in filtered_df.columns:
                    st.subheader("Trends Over Time")
                    fig3 = px.line(
                        filtered_df.groupby('Vintage Year').agg({
                            'Gross IRR': 'mean',
                            'Gross EM': 'mean',
                            'Co-Invest Equity': 'sum'
                        }).reset_index(),
                        x='Vintage Year',
                        y=['Gross IRR', 'Gross EM'],
                        secondary_y=['Co-Invest Equity'],
                        labels={'value': 'Metric Value'}
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            
            with tab3:
                # Export options
                st.subheader("Export Options")
                
                export_format = st.radio(
                    "Select Export Format",
                    ["CSV", "Excel", "JSON"],
                    horizontal=True
                )
                
                if export_format == "CSV":
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download as CSV",
                        data=csv,
                        file_name="pipeline_data.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel_buffer = BytesIO()
                    filtered_df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        "Download as Excel",
                        data=excel_buffer.getvalue(),
                        file_name="pipeline_data.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    json_data = filtered_df.to_json(orient="records")
                    st.download_button(
                        "Download as JSON",
                        data=json_data,
                        file_name="pipeline_data.json",
                        mime="application/json"
                    )
        else:
            st.warning("No deals match your filter criteria")
        
        st.button("← Return to Home", on_click=go_home, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing pipeline data: {str(e)}")
        st.error("Please ensure your file has the correct format with a 'Pipeline' sheet")
def render_market_research():
    st.title("Market Research")

    st.subheader("Townsend Views")

    # Embed Townsend Views page in a scrollable iframe
    components.iframe(
        src="https://www.townsendgroup.com/townsend-views/",
        width=1000,  # You can adjust width
        height=600,  # You can adjust height
        scrolling=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Return to Home button
    st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_market_research")


def render_smart_benchmarks():
    # Define benchmark data with categories and icons
    benchmarks = [
        {
            "category": "Townsend US Property Indices",
            "items": [
                {"name": "Townsend Core", "icon": "🏢", "file": "B1_Core.xlsx"},
                {"name": "Townsend Non Core", "icon": "🏗️", "file": "B2_NonCore.xlsx"},
                {"name": "Townsend Value Add", "icon": "🔄", "file": "B3_ValueAdd.xlsx"},
                {"name": "Townsend Opportunistic", "icon": "🎯", "file": "B4_Opportunistic.xlsx"},
            ]
        },
        {
            "category": "Market Segment Indices",
            "items": [
                {"name": "Townsend Majors", "icon": "📊", "note": "(True Market)"},
                {"name": "Townsend Expanded Market", "icon": "🌐", "note": "(All Stocks)"},
                {"name": "Townsend Minors", "icon": "📈", "note": "(Small Cap / Mid Cap)"},
                {"name": "Townsend Sector Specific", "icon": "🏘️", "note": "(Property Sector Focused Indices)"},
            ]
        },
        {
            "category": "Global Indices",
            "items": [
                {"name": "Townsend Global Property Index", "icon": "🌍"},
                {"name": "Townsend EMEA Property Index", "icon": "🇪🇺"},
                {"name": "Townsend APAC Property Index", "icon": "🌏"},
                {"name": "Townsend Global Infrastructure Index", "icon": "🛣️", "note": "(New Index)"},
                {"name": "Townsend Global Real Assets Index", "icon": "🏦", "note": "(Combine Global Infra and True Market)"},
            ]
        }
    ]

    # Handle selected benchmark view
    if "selected_benchmark" in st.session_state and st.session_state.selected_benchmark:
        selected = st.session_state.selected_benchmark
        file = next(
            (item["file"] for category in benchmarks for item in category["items"] 
             if item.get("file") and item["name"] == selected),
            None
        )
        
        st.title(selected)
        
        if file:
            try:
                df = pd.read_excel(BASE_DIR / file)
                
                # Format percentage columns
                percent_cols = [col for col in df.columns if any(x in col.lower() for x in ['return', 'growth', 'rate'])]
                for col in percent_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
                
                # Display data with tabs
                tab1, tab2 = st.tabs(["Data Table", "Visualization"])
                
                with tab1:
                    st.dataframe(df, use_container_width=True, height=600)
                
                with tab2:
                    # Add simple visualization
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_col = st.selectbox("Select metric to visualize", numeric_cols)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        if len(df) > 10:  # For large datasets, use barh
                            df.nlargest(10, selected_col).plot.barh(y=selected_col, ax=ax)
                        else:
                            df.plot.bar(x=df.columns[0], y=selected_col, ax=ax)
                        ax.set_title(f"{selected} - {selected_col}")
                        st.pyplot(fig)
                    else:
                        st.warning("No numeric columns available for visualization")
                
                # Add download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"{selected.replace(' ', '_')}.csv",
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Failed to load benchmark data: {e}")
        else:
            # For benchmarks without data files
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    st.image(Image.open(BASE_DIR / f"{selected.split()[-1]}.png"), width=200)
                except:
                    st.warning("Image not available")
            with col2:
                st.subheader("Benchmark Overview")
                category_name = next(
                    (cat['category'] for cat in benchmarks 
                     if any(item['name'] == selected for item in cat['items'])),
                    ''
                )
                st.markdown(f"""
                - **Category**: {category_name}
                - **Status**: Active
                - **Coverage**: Global
                """)
                st.info("Detailed benchmark data coming soon. Check back later for updates.")
        
        st.button("← Back to Benchmarks", 
                 on_click=lambda: st.session_state.pop("selected_benchmark"), 
                 use_container_width=True)
        return

    # Main benchmark selection interface
    st.title("Smart Benchmarks")
    st.markdown("Explore Townsend's proprietary benchmark indices across different market segments")
    
    for category in benchmarks:
        with st.expander(f"### {category['category']}", expanded=True):
            cols = st.columns(3)
            for idx, item in enumerate(category["items"]):
                with cols[idx % 3]:
                    container = st.container(border=True)
                    container.markdown(f"#### {item.get('icon', '📌')} {item['name']}")
                    if item.get("note"):
                        container.caption(item["note"])
                    
                    if item.get("file"):
                        if container.button("View Details", 
                                           key=f"btn_{item['name']}", 
                                           use_container_width=True):
                            st.session_state.selected_benchmark = item["name"]
                            st.rerun()
                    else:
                        container.info("Coming soon")
    
    st.button("← Return to Home", on_click=go_home, use_container_width=True)

def main():
    # Initialize session state values if not set
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "scenario" not in st.session_state:
        st.session_state.scenario = None
    if "selected_benchmark" not in st.session_state:
        st.session_state.selected_benchmark = None

    if st.session_state.page == "home":
        landing_page()
    elif st.session_state.page.startswith("option"):
        option_num = st.session_state.page[-1]
        render_option(option_num)

if __name__ == "__main__":
    main()
