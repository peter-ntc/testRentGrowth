
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# App state setup
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

        
        

        # Add Return to Home button at bottom of page
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True)

        
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

        # Return to Home button at bottom
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True)

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

        with row4_left:
            st.button("Compare ALL 3 scenarios", on_click=set_scenario, args=("compare",), key="btn_compare_main", use_container_width=True)
        with row4_right:
            # Show comparison chart beside button
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

            st.button("Compare ALL 3 scenarios", on_click=set_scenario, args=("compare",), key="btn_compare_main", use_container_width=True)

    else:
        label_map = {
            "consensus": "Consensus Economic Outlook",
            "high": "Higher Growth & Inflation",
            "low": "Lower Growth & Inflation",
            "compare": "Compare ALL 3 scenarios"
        }
        label = st.session_state.scenario
        st.subheader(label_map.get(label, "Scenario"))

        if label == "consensus":
            df = safe_load_df("BaseScenario.xlsx", label_map[label])
            if df is not None:
                plot_chart(df, label_map[label])
            render_scenario_table("BaseScenario.xlsx", key_prefix="base")
        elif label == "high":
            df = safe_load_df("HighScenario.xlsx", label_map[label])
            if df is not None:
                plot_chart(df, label_map[label])
            render_scenario_table("HighScenario.xlsx", key_prefix="high")
        elif label == "low":
            df = safe_load_df("LowScenario.xlsx", label_map[label])
            if df is not None:
                plot_chart(df, label_map[label])
            render_scenario_table("LowScenario.xlsx", key_prefix="low")
        else:
            st.markdown("🚧 Under Construction 🚧")

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True)

def render_option(option_num):
    if option_num == "1":
        render_forecasting_modeling()
    else:
        option_names = [
            "Forecasting & Modeling",
            "Optimizer",
            "Fund & Deal Pipeline",
            "Smart Benchmarks",
            "Secondaries Marketplace",
            "Market Research"
        ]
        st.title(option_names[int(option_num)-1])
        st.subheader("🚧 Under Construction 🚧")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True)

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
            st.button(option_labels[i], on_click=set_page, args=(i+1,))

    col4, col5, col6 = st.columns(3)
    for i, col in enumerate([col4, col5, col6], start=3):
        with col:
            st.button(option_labels[i], on_click=set_page, args=(i+1,))

def main():
    if st.session_state.page == "home":
        landing_page()
    elif st.session_state.page.startswith("option"):
        option_num = st.session_state.page[-1]
        render_option(option_num)

if __name__ == "__main__":
    main()
