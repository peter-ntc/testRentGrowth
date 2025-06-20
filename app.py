import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
from pathlib import Path

# Setup base path
BASE_DIR = Path(__file__).parent

# Load logo
logo = Image.open(BASE_DIR / "townsendAI_logo_1.png")

# Initialize session state
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

def render_consensus_table():
    try:
        df = pd.read_excel(BASE_DIR / "BaseScenario.xlsx", header=None, usecols="C,D,K,L,N,O,P,Q,R,S,V,W,AE,AF", skiprows=2, nrows=19)
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
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading table: {e}")

def render_forecasting_modeling():
    st.title("Forecasting & Modeling")
    st.markdown("### Forecasting and Modeling supported for 3 economic scenarios.")

    if st.session_state.scenario is None:
        col_left, col_right = st.columns([1, 3])
        with col_left:
            st.button("Consensus Economic Outlook", on_click=set_scenario, args=("consensus",), use_container_width=True)
            st.button("Higher Growth & Inflation", on_click=set_scenario, args=("high",), use_container_width=True)
            st.button("Lower Growth & Inflation", on_click=set_scenario, args=("low",), use_container_width=True)
            st.button("Compare ALL 3 scenarios", on_click=set_scenario, args=("compare",), use_container_width=True)

        with col_right:
            df_base = safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook")
            df_high = safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation")
            df_low = safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation")

            if df_base is not None:
                plot_chart(df_base, "Consensus Economic Outlook")
            if df_high is not None:
                plot_chart(df_high, "Higher Growth & Inflation")
            if df_low is not None:
                plot_chart(df_low, "Lower Growth & Inflation")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Consensus Economic Outlook", on_click=set_scenario, args=("consensus",), use_container_width=True)
            st.button("Higher Growth & Inflation", on_click=set_scenario, args=("high",), use_container_width=True)
        with col2:
            st.button("Lower Growth & Inflation", on_click=set_scenario, args=("low",), use_container_width=True)
            st.button("Smart Benchmarks", on_click=set_scenario, args=("benchmarks",), use_container_width=True)

        # Show all 3 scenario charts here
        df_base = safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook")
        df_high = safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation")
        df_low = safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation")

        if df_base is not None:
            plot_chart(df_base, "Consensus Economic Outlook")
        if df_high is not None:
            plot_chart(df_high, "Higher Growth & Inflation")
        if df_low is not None:
            plot_chart(df_low, "Lower Growth & Inflation")
    else:
        label_map = {
            "consensus": "Consensus Economic Outlook",
            "high": "Higher Growth & Inflation",
            "low": "Lower Growth & Inflation",
            "benchmarks": "Smart Benchmarks"
        }
        label = st.session_state.scenario
        st.subheader(label_map.get(label, "Scenario"))

        if label == "consensus":
            df = safe_load_df("BaseScenario.xlsx", label_map[label])
            if df is not None:
                plot_chart(df, label_map[label])
            render_consensus_table()
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
