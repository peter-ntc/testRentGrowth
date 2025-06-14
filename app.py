
import streamlit as st
import pandas as pd
import plotly.express as px

# Title and description
st.title("Rent Growth Forecast Comparison")
st.markdown("""
This dashboard compares **Rent Growth Forecasts** across three economic scenarios:
**Base**, **High**, and **Low** for:

- Core Plus – Res
- Core Plus – Ind
""")

# Load scenario data
@st.cache_data
def load_data():
    base = pd.read_excel("BaseScenario.xlsx")
    high = pd.read_excel("HighScenario.xlsx")
    low = pd.read_excel("LowScenario.xlsx")

    col_range = list(range(13, 19))  # Excel columns N to S (0-indexed)
    row_labels = ['Core Plus - Res', 'Core Plus - Ind']
    row_indices = [4, 5]  # Rows 5 and 6 (0-indexed)

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

df = load_data()

# Plot
fig = px.bar(
    df,
    x="Period",
    y="Rent Growth",
    color="Scenario",
    barmode="group",
    facet_row="index",
    title="Rent Growth Forecasts Across Scenarios"
)

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)
