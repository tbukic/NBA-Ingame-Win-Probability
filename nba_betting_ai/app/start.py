import streamlit as st

page = st.navigation([
    # st.Page("live_scores.py", title="Live Scores", icon="🏀"),
    st.Page("inference.py", title="Prediction", icon="📈"),
    # st.Page("legacy_inference.py", title="Legacy Prediction", icon="📊"),
])
page.run()