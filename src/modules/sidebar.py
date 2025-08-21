import streamlit as st
from config.settings import HOME_FILE, INFERENCE_FILE, DATASET_FILE, TRAIN_FILE

def render_sidebar():
    with st.sidebar:
        st.markdown("## Quick Navigation")
        st.page_link(HOME_FILE, label="Home", icon="🤟")
        st.page_link(INFERENCE_FILE, label="Realtime Simulation", icon="▶")
        st.page_link(DATASET_FILE, label="Create Dataset", icon="📸")
        st.page_link(TRAIN_FILE, label="Train Custom Model", icon="🧠")