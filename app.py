from capm_module import parse_capm_input, run_capm_optimizer, plot_capm_frontier, write_capm_output


import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
            st.subheader("CAPM Optimizer")

            uploaded_file = st.file_uploader("Upload CAPM Input File (capm_input.xlsx)", type=["xlsx"], key="capm_upload")
            if uploaded_file:
                try:
                    sectors, exp_ret, vol, corr, wmin, wmax, rf = parse_capm_input(uploaded_file)
                    df_result, max_idx, frontier_risks, target_returns, sim_risks, sim_returns, sim_sharpes = run_capm_optimizer(
                        sectors, exp_ret, vol, corr, wmin, wmax, rf)

                    fig = plot_capm_frontier(df_result, max_idx, frontier_risks, target_returns, sim_risks, sim_returns, sim_sharpes)
                    st.pyplot(fig)

                    st.markdown("### Optimized Frontier Table")
                    st.dataframe(df_result.style.format({col: "{:.2%}" for col in df_result.columns if col != "Sharpe Ratio"}), use_container_width=True)

                    output = write_capm_output(df_result)
                    st.download_button("📥 Download CAPM Output", output, file_name="capm_output.xlsx", use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_capm")
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
    else:
        option_labels = [
            "Forecasting & Modeling",
            "Optimizer",
            "Fund & Deal Pipeline",
            "Smart Benchmarks",
            "Secondaries Marketplace",
            "Market Research"
        ]
        st.title(option_labels[int(option_num)-1])
        st.subheader("🚧 Under Construction 🚧")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key=f"btn_return_option{option_num}")
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

def main():
    if st.session_state.page == "home":
        landing_page()
    

elif st.session_state.scenario == "capm":
    st.subheader("CAPM Optimizer")

    uploaded_file = st.file_uploader("Upload CAPM Input File (capm_input.xlsx)", type=["xlsx"], key="capm_upload")
    if uploaded_file:
        try:
            sectors, exp_ret, vol, corr, wmin, wmax, rf = parse_capm_input(uploaded_file)
            df_result, max_idx, frontier_risks, target_returns, sim_risks, sim_returns, sim_sharpes = run_capm_optimizer(
                sectors, exp_ret, vol, corr, wmin, wmax, rf)

            # Plot and display
            fig = plot_capm_frontier(df_result, max_idx, frontier_risks, target_returns, sim_risks, sim_returns, sim_sharpes)
            st.pyplot(fig)

            st.markdown("### Optimized Frontier Table")
            st.dataframe(df_result.style.format({col: "{:.2%}" for col in df_result.columns if col != "Sharpe Ratio"}), use_container_width=True)

            # Downloadable file
            output = write_capm_output(df_result)
            st.download_button("📥 Download CAPM Output", output, file_name="capm_output.xlsx", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔙 Return to Home", on_click=go_home, use_container_width=True, key="btn_return_capm")


elif st.session_state.page.startswith("option"):
        option_num = st.session_state.page[-1]
        render_option(option_num)

if __name__ == "__main__":
    main()
