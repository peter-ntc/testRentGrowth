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
    elif option_num == "2" or option_num == "3" or option_num == "4":
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
            if label == "Consensus Economic Outlook":
                st.markdown("### Sector Summary Table")
                st.markdown("""<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sector</th>
      <th>Entry Cap Rate</th>
      <th>Exit Cap Rate Year 3</th>
      <th>Exit Cap Rate Year 6</th>
      <th>Rent Growth Year 1</th>
      <th>Rent Growth Year 2</th>
      <th>Rent Growth Year 3</th>
      <th>Rent Growth Year 4</th>
      <th>Rent Growth Year 5</th>
      <th>Rent Growth Year 6</th>
      <th>3 Yr Unlevered Property Returns</th>
      <th>6 Yr Unlevered Property Returns</th>
      <th>3 Yr Levered Fund Net Returns</th>
      <th>6 Yr Levered Fund Net Returns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Core - Res</td>
      <td>4.60%</td>
      <td>2.97%</td>
      <td>4.76%</td>
      <td>-1.20%</td>
      <td>1.00%</td>
      <td>2.40%</td>
      <td>4.00%</td>
      <td>5.00%</td>
      <td>2.50%</td>
      <td>19.86%</td>
      <td>5.60%</td>
      <td>19.36%</td>
      <td>5.05%</td>
    </tr>
    <tr>
      <td>Core - Ind</td>
      <td>3.70%</td>
      <td>3.48%</td>
      <td>4.64%</td>
      <td>6.10%</td>
      <td>5.00%</td>
      <td>4.30%</td>
      <td>4.00%</td>
      <td>3.50%</td>
      <td>3.00%</td>
      <td>10.60%</td>
      <td>4.16%</td>
      <td>10.35%</td>
      <td>3.73%</td>
    </tr>
    <tr>
      <td>Core - Off</td>
      <td>5.60%</td>
      <td>9.92%</td>
      <td>7.79%</td>
      <td>-1.80%</td>
      <td>-1.10%</td>
      <td>-0.80%</td>
      <td>0.50%</td>
      <td>1.00%</td>
      <td>2.00%</td>
      <td>-13.63%</td>
      <td>-0.97%</td>
      <td>-14.63%</td>
      <td>-3.96%</td>
    </tr>
    <tr>
      <td>Core - Retail</td>
      <td>5.50%</td>
      <td>6.16%</td>
      <td>5.54%</td>
      <td>1.00%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>2.50%</td>
      <td>2.60%</td>
      <td>6.32%</td>
      <td>1.85%</td>
      <td>5.63%</td>
    </tr>
    <tr>
      <td>Core-Plus - Res</td>
      <td>4.80%</td>
      <td>2.97%</td>
      <td>4.76%</td>
      <td>-1.60%</td>
      <td>0.60%</td>
      <td>2.30%</td>
      <td>4.00%</td>
      <td>5.00%</td>
      <td>2.50%</td>
      <td>21.26%</td>
      <td>6.25%</td>
      <td>21.01%</td>
      <td>6.38%</td>
    </tr>
    <tr>
      <td>Core-Plus - Ind</td>
      <td>4.10%</td>
      <td>4.06%</td>
      <td>5.22%</td>
      <td>6.60%</td>
      <td>5.90%</td>
      <td>5.00%</td>
      <td>3.50%</td>
      <td>3.00%</td>
      <td>2.50%</td>
      <td>9.92%</td>
      <td>4.46%</td>
      <td>9.42%</td>
      <td>3.80%</td>
    </tr>
    <tr>
      <td>Core-Plus - Off</td>
      <td>5.40%</td>
      <td>9.92%</td>
      <td>7.79%</td>
      <td>-1.40%</td>
      <td>-0.70%</td>
      <td>-0.30%</td>
      <td>0.50%</td>
      <td>1.00%</td>
      <td>2.00%</td>
      <td>-14.35%</td>
      <td>-1.40%</td>
      <td>-15.60%</td>
      <td>-5.92%</td>
    </tr>
    <tr>
      <td>Core-Plus - Ret</td>
      <td>5.90%</td>
      <td>6.16%</td>
      <td>5.54%</td>
      <td>1.00%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>2.50%</td>
      <td>5.15%</td>
      <td>7.72%</td>
      <td>4.15%</td>
      <td>8.24%</td>
    </tr>
    <tr>
      <td>Data Center</td>
      <td>5.40%</td>
      <td>3.09%</td>
      <td>5.56%</td>
      <td>5.90%</td>
      <td>5.90%</td>
      <td>5.10%</td>
      <td>5.00%</td>
      <td>4.00%</td>
      <td>3.00%</td>
      <td>31.13%</td>
      <td>9.00%</td>
      <td>30.88%</td>
      <td>10.36%</td>
    </tr>
    <tr>
      <td>Single-Family Rental</td>
      <td>4.50%</td>
      <td>3.47%</td>
      <td>4.63%</td>
      <td>4.80%</td>
      <td>4.80%</td>
      <td>4.90%</td>
      <td>4.00%</td>
      <td>4.00%</td>
      <td>3.00%</td>
      <td>18.05%</td>
      <td>7.90%</td>
      <td>17.55%</td>
      <td>8.00%</td>
    </tr>
    <tr>
      <td>Cold Storage</td>
      <td>3.80%</td>
      <td>4.22%</td>
      <td>6.63%</td>
      <td>10.00%</td>
      <td>8.00%</td>
      <td>5.00%</td>
      <td>4.00%</td>
      <td>3.00%</td>
      <td>2.00%</td>
      <td>7.54%</td>
      <td>0.35%</td>
      <td>6.79%</td>
      <td>-2.20%</td>
    </tr>
    <tr>
      <td>Senior Housing</td>
      <td>5.20%</td>
      <td>3.90%</td>
      <td>5.19%</td>
      <td>8.50%</td>
      <td>6.10%</td>
      <td>5.10%</td>
      <td>4.00%</td>
      <td>4.00%</td>
      <td>3.00%</td>
      <td>21.26%</td>
      <td>9.45%</td>
      <td>20.51%</td>
      <td>10.69%</td>
    </tr>
    <tr>
      <td>Student Housing</td>
      <td>5.75%</td>
      <td>5.06%</td>
      <td>5.62%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>2.50%</td>
      <td>2.50%</td>
      <td>2.00%</td>
      <td>12.57%</td>
      <td>8.30%</td>
      <td>11.82%</td>
      <td>8.64%</td>
    </tr>
    <tr>
      <td>Affordable Housing</td>
      <td>5.60%</td>
      <td>4.30%</td>
      <td>4.30%</td>
      <td>3.00%</td>
      <td>3.50%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>17.60%</td>
      <td>12.57%</td>
      <td>16.85%</td>
      <td>14.75%</td>
    </tr>
    <tr>
      <td>Manufactured Home Park</td>
      <td>4.00%</td>
      <td>4.72%</td>
      <td>4.72%</td>
      <td>5.00%</td>
      <td>5.00%</td>
      <td>4.00%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>3.00%</td>
      <td>2.79%</td>
      <td>4.81%</td>
      <td>2.04%</td>
      <td>3.71%</td>
    </tr>
    <tr>
      <td>Medical Office</td>
      <td>5.60%</td>
      <td>5.17%</td>
      <td>5.52%</td>
      <td>2.50%</td>
      <td>2.50%</td>
      <td>2.50%</td>
      <td>2.50%</td>
      <td>2.30%</td>
      <td>2.20%</td>
      <td>10.12%</td>
      <td>7.64%</td>
      <td>9.37%</td>
      <td>7.70%</td>
    </tr>
    <tr>
      <td>Life Science</td>
      <td>5.30%</td>
      <td>5.81%</td>
      <td>5.81%</td>
      <td>3.90%</td>
      <td>2.80%</td>
      <td>2.40%</td>
      <td>2.50%</td>
      <td>2.50%</td>
      <td>2.50%</td>
      <td>4.77%</td>
      <td>6.08%</td>
      <td>3.77%</td>
      <td>5.12%</td>
    </tr>
    <tr>
      <td>Self-Storage</td>
      <td>4.40%</td>
      <td>5.16%</td>
      <td>5.71%</td>
      <td>-1.50%</td>
      <td>2.30%</td>
      <td>2.70%</td>
      <td>2.50%</td>
      <td>2.00%</td>
      <td>2.00%</td>
      <td>0.16%</td>
      <td>1.86%</td>
      <td>-0.59%</td>
      <td>-0.51%</td>
    </tr>
    <tr>
      <td>IOS</td>
      <td>3.70%</td>
      <td>3.76%</td>
      <td>5.91%</td>
      <td>7.00%</td>
      <td>6.00%</td>
      <td>5.00%</td>
      <td>4.00%</td>
      <td>3.00%</td>
      <td>2.00%</td>
      <td>9.09%</td>
      <td>1.12%</td>
      <td>8.59%</td>
      <td>-1.01%</td>
    </tr>
  </tbody>
</table>""", unsafe_allow_html=True)

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
