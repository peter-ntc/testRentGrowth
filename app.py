
import streamlit as st
import pandas as pd
from capm_optimizer import run_capm_optimizer

def main():
    st.image("townsendAI_logo_1.png", width=100)
    st.title("📈 CAPM Optimization")
    st.markdown("### Efficient Frontier")

    try:
        output = run_capm_optimizer()
        st.image(output["frontier"], caption="Efficient Frontier", use_container_width=True)
        st.dataframe(output["optimized_weights"])
        st.download_button("📥 Download Results", data=output["excel"], file_name="capm_output.xlsx")
    except Exception as e:
        st.error(f"Error running optimizer: {e}")

if __name__ == "__main__":
    main()
