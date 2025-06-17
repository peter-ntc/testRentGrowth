
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="centered", page_title="TownsendAI Forecasting Tool")

def render_clickable_logo(image_path, url, width=120):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    html = f'''
        <div style="text-align:center; margin-bottom: 1rem;">
            <a href="{url}" target="_blank">
                <img src="data:image/png;base64,{img_str}" width="{width}"/>
            </a>
        </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

render_clickable_logo("townsendAI_logo 1.png", "https://townsendgroup.com")

st.title("🏢 TownsendAI Forecasting Tool")
st.markdown("Powered by 40+ years of proprietary real estate data and forecasting models.")

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

# --- Session state setup ---
if "last_forecast_type" not in st.session_state:
    st.session_state.last_forecast_type = ""

forecast_type = st.selectbox("Select Forecast Type", ["", "Rent Growth", "Return Forecast"], key="forecast_type")

# Reset session values if forecast type changes
if forecast_type != st.session_state.last_forecast_type:
    st.session_state.last_forecast_type = forecast_type
    st.session_state["scenario"] = ""
    st.session_state["sectors"] = []

scenario = st.selectbox("Select Scenario", ["", "Base", "High", "Low", "All"], key="scenario")

sectors = st.multiselect(
    "Select up to 3 sectors:",
    get_sectors(),
    key="sectors",
    max_selections=3
)

if forecast_type and scenario and sectors:
    data = extract_data(scenario, sectors, forecast_type)
    df = convert_df(data)

    view_option = st.radio("Choose Display Type", ["Charts", "Table"], horizontal=True)

    if view_option == "Table":
        st.dataframe(df)

    elif view_option == "Charts":
        if forecast_type == "Rent Growth":
            st.subheader("📊 Rent Growth Forecasts")
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
            st.subheader("📊 6-Year Returns (Unlevered and Levered)")
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

    # Download Options
    st.markdown("### 📥 Download Data")
    excel_bytes = BytesIO()
    df.to_excel(excel_bytes, index=False, engine='openpyxl')
    st.download_button("Download as Excel", data=excel_bytes.getvalue(), file_name="forecast.xlsx")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv, file_name="forecast.csv")

    try:
        import pdfkit
        pdf_bytes = BytesIO()
        df.to_html(buf=pdf_bytes)
        st.download_button("Download as PDF", data=pdf_bytes.getvalue(), file_name="forecast.html")
    except:
        st.info("PDF export requires `pdfkit` and wkhtmltopdf. Skipped for now.")
