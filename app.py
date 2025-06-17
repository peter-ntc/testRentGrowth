
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(page_title="TownsendAI", layout="centered")

def render_clickable_logo(image_path, url, width=120):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    html = f'''
        <div style="text-align:center;">
            <a href="{url}" target="_blank">
                <img src="data:image/png;base64,{img_str}" width="{width}" style="margin-bottom: 1rem;" />
            </a>
        </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def show_back_button():
    if st.button("🔙 Back to Home", key=f"back_{st.session_state.page}"):
        st.session_state.page = "home"
        st.experimental_rerun()

if "page" not in st.session_state:
    st.session_state.page = "home"

render_clickable_logo("townsendAI_logo 1.png", "https://townsendgroup.com")

def home():
    st.title("🏢 TownsendAI Platform")
    st.markdown("### Explore our tools:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Forecasting & Modeling"):
            st.session_state.page = "forecasting"
        if st.button("🧠 Smart Benchmarks"):
            st.session_state.page = "benchmarks"
    with col2:
        if st.button("⚙️ Optimizer"):
            st.session_state.page = "optimizer"
        if st.button("📊 Secondaries Marketplace"):
            st.session_state.page = "secondaries"
    with col3:
        if st.button("💼 Fund & Deal Pipeline"):
            st.session_state.page = "pipeline"
        if st.button("📰 Market Research"):
            st.session_state.page = "research"

def coming_soon(section):
    st.title(section)
    show_back_button()
    st.info("🚧 This section is under construction. Stay tuned!")

def forecasting_page():
    st.title("📈 Forecasting & Modeling")
    show_back_button()

    excel_files = {
        "Base": "BaseScenario.xlsx",
        "High": "HighScenario.xlsx",
        "Low": "LowScenario.xlsx"
    }

    @st.cache_data
    def get_sectors():
        df = pd.read_excel(excel_files["Base"], sheet_name=0, header=None)
        return df.loc[2:20, 2].dropna().tolist()

    def extract_data(scenario, sectors, data_type):
        all_data = {}
        scenario_list = [scenario] if scenario != "All" else list(excel_files.keys())
        for scen in scenario_list:
            df = pd.read_excel(excel_files[scen], sheet_name=0, header=None)
            sector_names = df.loc[2:20, 2].tolist()
            for sector in sectors:
                try:
                    idx = sector_names.index(sector) + 2
                    if data_type == "Rent Growth":
                        values = df.loc[idx, 13:18].values.tolist()
                        cols = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"]
                    elif data_type == "Return Forecast":
                        values = [df.loc[idx, 22], df.loc[idx, 31]]
                        cols = ["Unlevered", "Levered"]
                    all_data.setdefault(sector, {})[scen] = dict(zip(cols, values))
                except ValueError:
                    continue
        return all_data

    def convert_df(data_dict):
        records = []
        for sector, scenario_values in data_dict.items():
            for scen, value_dict in scenario_values.items():
                row = {"Sector": sector, "Scenario": scen}
                row.update(value_dict)
                records.append(row)
        return pd.DataFrame(records)

    forecast_type = st.selectbox("Select Forecast Type", ["", "Rent Growth", "Return Forecast"])
    if forecast_type:
        scenario = st.selectbox("Select Scenario", ["", "Base", "High", "Low", "All"])
        sectors = st.multiselect("Select up to 3 sectors:", get_sectors(), max_selections=3)
        if scenario and sectors:
            data = extract_data(scenario, sectors, forecast_type)
            df = convert_df(data)
            view_option = st.radio("Choose Display Type", ["Charts", "Table"], horizontal=True)
            if view_option == "Table":
                st.dataframe(df)
            elif view_option == "Charts":
                if forecast_type == "Rent Growth":
                    years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"]
                    for sector, scenario_values in data.items():
                        fig, ax = plt.subplots()
                        bar_width = 0.2
                        x = list(range(len(years)))
                        for i, (scen, values) in enumerate(scenario_values.items()):
                            offset = [xi + (i - 1) * bar_width for xi in x]
                            ax.bar(offset, list(values.values()), width=bar_width, label=scen)
                        ax.set_xticks(x)
                        ax.set_xticklabels(years)
                        ax.set_title(f"{sector} - Rent Growth")
                        ax.set_ylabel("% Growth")
                        ax.legend()
                        st.pyplot(fig)
                elif forecast_type == "Return Forecast":
                    for sector, scenario_values in data.items():
                        fig, ax = plt.subplots()
                        width = 0.35
                        x = list(range(len(scenario_values)))
                        scenarios = list(scenario_values.keys())
                        unlevered = [scenario_values[scen]["Unlevered"] for scen in scenarios]
                        levered = [scenario_values[scen]["Levered"] for scen in scenarios]
                        ax.bar([i - width/2 for i in x], unlevered, width=width, label="Unlevered")
                        ax.bar([i + width/2 for i in x], levered, width=width, label="Levered")
                        ax.set_xticks(x)
                        ax.set_xticklabels(scenarios)
                        ax.set_title(f"{sector} - Return Forecast")
                        ax.set_ylabel("% Return")
                        ax.legend()
                        st.pyplot(fig)
            st.markdown("### 📥 Download Data")
            excel_bytes = BytesIO()
            df.to_excel(excel_bytes, index=False, engine='openpyxl')
            st.download_button("Download as Excel", data=excel_bytes.getvalue(), file_name="forecast.xlsx")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download as CSV", data=csv, file_name="forecast.csv")

# Router
if st.session_state.page == "home":
    home()
elif st.session_state.page == "forecasting":
    forecasting_page()
elif st.session_state.page == "optimizer":
    coming_soon("⚙️ Optimizer")
elif st.session_state.page == "pipeline":
    coming_soon("💼 Fund & Deal Pipeline")
elif st.session_state.page == "benchmarks":
    coming_soon("🧠 Smart Benchmarks")
elif st.session_state.page == "secondaries":
    coming_soon("📊 Secondaries Marketplace")
elif st.session_state.page == "research":
    coming_soon("📰 Market Research")
