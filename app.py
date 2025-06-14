
import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Rent Growth Forecast Comparison")

st.markdown("""
This dashboard allows you to compare **Rent Growth Forecasts** across different economic scenarios:

- Core Plus – Res
- Core Plus – Ind

Choose a scenario below to filter the data.
""")

# Load scenario data
@st.cache_data
def load_data():
    base = pd.read_excel("BaseScenario.xlsx")
    high = pd.read_excel("HighScenario.xlsx")
    low = pd.read_excel("LowScenario.xlsx")

    col_range = list(range(13, 19))  # Columns N to S
    row_labels = ['Core Plus - Res', 'Core Plus - Ind']
    row_indices = [4, 5]

    def extract(df, scenario):
        temp = df.iloc[row_indices, col_range]
        temp.index = row_labels
        temp.columns = [f"Period {i+1}" for i in range(temp.shape[1])]
        melted = temp.reset_index().melt(id_vars='index', var_name='Period', value_name='Rent Growth')
        melted['Scenario'] = scenario
        return melted

    return pd.concat([
        extract(base, 'Base'),
        extract(high, 'High'),
        extract(low, 'Low')
    ])

# Load data
df = load_data()

# User input
scenario_choice = st.selectbox(
    "Select a scenario to display:",
    options=["All", "Base", "High", "Low"]
)

# Filter data
if scenario_choice != "All":
    df = df[df["Scenario"] == scenario_choice]

# Plot
fig = px.bar(
    df,
    x="Period",
    y="Rent Growth",
    color="Scenario" if scenario_choice == "All" else None,
    barmode="group",
    facet_row="index",
    title=f"Rent Growth Forecasts - {scenario_choice} Scenario" if scenario_choice != "All" else "Rent Growth Forecasts - All Scenarios"
)

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)
