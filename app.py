import streamlit as st
from PIL import Image

# Load logo
logo = Image.open("townsendAI_logo_1.png")

# Define routing state
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def render_option(option_num):
    st.title(f"Option {option_num}")
    st.subheader("🚧 Under Construction 🚧")
    if st.button("🔙 Return to Home"):
        go_home()

def landing_page():
    st.image(logo, use_column_width=False)
    st.title("TownsendAI")
    st.write("Welcome to the MVP. Please select an option:")

    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3], start=1):
        if col.button(f"Option {i}"):
            st.session_state.page = f"option{i}"

    col4, col5, col6 = st.columns(3)
    for i, col in enumerate([col4, col5, col6], start=4):
        if col.button(f"Option {i}"):
            st.session_state.page = f"option{i}"

# Main app controller
def main():
    if st.session_state.page == "home":
        landing_page()
    elif st.session_state.page.startswith("option"):
        option_num = st.session_state.page[-1]
        render_option(option_num)

if __name__ == "__main__":
    main()
