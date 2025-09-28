import streamlit as st

page = st.navigation([
    st.Page("inference.py", title="Prediction", icon="📈"),
])
page.run()