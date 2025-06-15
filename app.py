
import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Rent Growth Forecast Comparison")

st.markdown("""
This dashboard lets you explore **Rent Growth Forecasts** across up to 3 sectors and compare across scenarios.

- First, select up to 3 sectors
- Then choose a scenario to display
- The chart will appear once both are selected
""")

# Load scenario data
@st.cache_data
def load_data():
    base = pd.read_excel("BaseScenario.xlsx")
    high = pd.read_excel("HighScenario.xlsx")
    low = pd.read_excel("LowScenario.xlsx")

    col_range = list(range(13, 19))  # Columns N to S
    all_rows = base.iloc[:, 2].dropna().reset_index(drop=True)  # All profile names in column C

    def extract(df, scenario):
        temp = df.iloc[:, col_range]
        temp['Profile'] = df.iloc[:, 2]
        temp = temp.dropna(subset=['Profile'])
        temp = temp.reset_index(drop=True)
        temp = temp.set_index('Profile')
        temp.columns = [f"Period {i+1}" for i in range(temp.shape[1])]
        melted = temp.reset_index().melt(id_vars='Profile', var_name='Period', value_name='Rent Growth')
        melted['Scenario'] = scenario
        return melted

    return pd.concat([
        extract(base, 'Base'),
        extract(high, 'High'),
        extract(low, 'Low')
    ]), all_rows.tolist()

# Load data
df, all_profiles = load_data()

# Sector selection
selected_sectors = st.multiselect(
    "Select up to 3 sectors:",
    options=all_profiles,
    max_selections=3
)

# Scenario selection
scenario_choice = st.selectbox(
    "Select a scenario to display:",
    options=["", "Base", "High", "Low", "All"],
    index=0,
    format_func=lambda x: "Select..." if x == "" else x
)

# Only proceed if both selections are made
if selected_sectors and scenario_choice != "":
    # Filter by sectors
    filtered_df = df[df["Profile"].isin(selected_sectors)]

    # Filter by scenario
    if scenario_choice != "All":
        filtered_df = filtered_df[filtered_df["Scenario"] == scenario_choice]

    # Plot
    fig = px.bar(
        filtered_df,
        x="Period",
        y="Rent Growth",
        color="Scenario" if scenario_choice == "All" else None,
        barmode="group",
        facet_row="Profile",
        title=f"Rent Growth Forecast - {scenario_choice} Scenario" if scenario_choice != "All" else "Rent Growth Forecast - All Scenarios"
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select up to 3 sectors and choose a scenario to see the chart.")
