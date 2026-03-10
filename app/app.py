import streamlit as st

st.set_page_config(
    layout="centered",
    initial_sidebar_state="collapsed"
)

pg = st.navigation([
    st.Page("pages/home.py", title="AI Image Recognition", default=True)
], position="top")

pg.run()