import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
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
    return fig

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

    # Load Excel data using robust paths
    base = pd.read_excel(BASE_DIR / "BaseScenario.xlsx", header=None, usecols="C:I", skiprows=51, nrows=5)
    high = pd.read_excel(BASE_DIR / "HighScenario.xlsx", header=None, usecols="C:I", skiprows=51, nrows=5)
    low = pd.read_excel(BASE_DIR / "LowScenario.xlsx", header=None, usecols="C:I", skiprows=51, nrows=5)

    def clean(df):
        df_clean = df.iloc[1:4, 1:].copy()
        df_clean.columns = ["2025", "2026", "2027", "2028", "2029", "2030"]
        df_clean.index = ["GDP", "Inflation", "10 YR"]
        return df_clean.astype(float)

    base_data = clean(base)
    high_data = clean(high)
    low_data = clean(low)

    # Layout with side labels and charts
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.button("Consensus Economic Outlook", disabled=True, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Higher Growth & Inflation", disabled=True, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Lower Growth & Inflation", disabled=True, use_container_width=True)

    with right_col:
        st.pyplot(plot_chart(base_data, "Consensus Economic Outlook"))
        st.pyplot(plot_chart(high_data, "Higher Growth & Inflation"))
        st.pyplot(plot_chart(low_data, "Lower Growth & Inflation"))

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
