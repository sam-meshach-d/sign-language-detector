from pathlib import Path
import streamlit as st
from config.settings import ROOT_DIR

def load_css(file_name: str):
    css_path = ROOT_DIR / "static" / file_name
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
