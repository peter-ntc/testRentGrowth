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

# Define routing state
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def set_page(option):
    st.session_state.page = f"option{option}"

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

def safe_load_df(filename, scenario_label):
    try:
        df = pd.read_excel(BASE_DIR / filename, header=None, usecols="D:I", skiprows=52, nrows=3)
        if df.shape[1] != 6:
            st.warning(f"{scenario_label}: Expected 6 columns, found {df.shape[1]}. Please check Excel formatting.")
            return None
        df.columns = ["2025", "2026", "2027", "2028", "2029", "2030"]
        df.index = ["GDP", "Inflation", "10 YR"]
        return df.astype(float)
    except Exception as e:
        st.error(f"Failed to load {scenario_label} data: {e}")
        return None

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
        st.title(f"{option_names[int(option_num)-1]}")
        st.subheader("🚧 Under Construction 🚧")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔙 Return to Home", on_click=go_home, use_container_width=True)

def render_forecasting_modeling():
    st.title("Forecasting & Modeling")
    st.markdown("### Forecasting and Modeling supported for 3 economic scenarios.")

    scenarios = [
        ("Consensus Economic Outlook", "BaseScenario.xlsx"),
        ("Higher Growth & Inflation", "HighScenario.xlsx"),
        ("Lower Growth & Inflation", "LowScenario.xlsx")
    ]

    for label, file in scenarios:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.button(label, disabled=True, use_container_width=True, key=f"label_{str(uuid.uuid4())}")
        with col2:
            df = safe_load_df(file, label)
            if df is not None:
                plot_chart(df, label)

    # Comparison chart
    st.markdown("###")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.button("Comparison across scenarios", disabled=True, use_container_width=True, key="comparison_label")
    with col2:
        def get_avg(df, label):
            return df.loc[label].mean() if df is not None else 0.0

        avg_gdp = {
            "Consensus": get_avg(safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook"), "GDP"),
            "High": get_avg(safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation"), "GDP"),
            "Low": get_avg(safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation"), "GDP")
        }

        avg_inflation = {
            "Consensus": get_avg(safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook"), "Inflation"),
            "High": get_avg(safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation"), "Inflation"),
            "Low": get_avg(safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation"), "Inflation")
        }

        avg_10yr = {
            "Consensus": get_avg(safe_load_df("BaseScenario.xlsx", "Consensus Economic Outlook"), "10 YR"),
            "High": get_avg(safe_load_df("HighScenario.xlsx", "Higher Growth & Inflation"), "10 YR"),
            "Low": get_avg(safe_load_df("LowScenario.xlsx", "Lower Growth & Inflation"), "10 YR")
        }

        metrics = ["GDP", "Inflation", "10 YR"]
        consensus = [avg_gdp["Consensus"] * 100, avg_inflation["Consensus"] * 100, avg_10yr["Consensus"] * 100]
        high = [avg_gdp["High"] * 100, avg_inflation["High"] * 100, avg_10yr["High"] * 100]
        low = [avg_gdp["Low"] * 100, avg_inflation["Low"] * 100, avg_10yr["Low"] * 100]

        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, consensus, width, label="Consensus")
        ax.bar(x, high, width, label="High")
        ax.bar(x + width, low, width, label="Low")

        ax.set_ylabel("Average (%)")
        ax.set_title("Comparison of Average Metrics (2025–2030)")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

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

# Main app controller
def main():
    if st.session_state.page == "home":
        landing_page()
    elif st.session_state.page.startswith("option"):
        option_num = st.session_state.page[-1]
        render_option(option_num)

if __name__ == "__main__":
    main()
