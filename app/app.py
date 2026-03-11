import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    layout="centered",
    initial_sidebar_state="collapsed"
)

pg = st.navigation([
    st.Page("pages/home.py", title="AI Image Recognition", default=True)
], position="top")

pg.run()