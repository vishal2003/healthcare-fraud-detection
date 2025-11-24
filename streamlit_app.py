import streamlit as st 
from views import about_me, fraud_detection

# Set Streamlit page configuration (Title, favicon, layout)
st.set_page_config(page_title="Healthcare Claims Fraud Detection App", page_icon="💼", layout="wide")


# --- PAGE SETUP ---
PAGES = {
    "Health Claims Fraud Detection": fraud_detection,
    "About Me": about_me
}

# Sidebar Section for Navigation first
with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Go to", list(PAGES.keys()))

    st.title("Contact Me")
    st.write("📧 Email: [adesuadonatus45@gmail.com](mailto:adesuadonatus45@gmail.com)")
    st.write("🌐 Website: [Medium](https://medium.com/@adesua)")
    st.write("🔗 LinkedIn: [Ayomitan Adesua](https://www.linkedin.com/in/adesuaayomitandonatus/)")
    
# Load the selected page function
page = PAGES[selection]
page.app()  # Call the app function from the selected module 

