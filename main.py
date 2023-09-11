import streamlit as st

# Initialize session state variables
if 'openai_api_key' not in st.session_state:
	st.session_state.openai_api_key = ""

if 'serper_api_key' not in st.session_state:
	st.session_state.serper_api_key = ""

st.set_page_config(page_title="Home", page_icon="🦜️🔗")

st.header("Welcome to LangChain! 👋")