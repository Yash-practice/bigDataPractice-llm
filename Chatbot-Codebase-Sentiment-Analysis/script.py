from constants import domain_constant
from models import model
from page import main
import streamlit as st
import os
 
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

domains = [domain_constant.GENERAL, domain_constant.SOCIAL_MEDIA, domain_constant.CUSTOMER_REVIEW]
 
if __name__ == "__main__":
    
    st.set_page_config(layout="wide")
    
    # Load CSS
    css_file = 'css/style.css'
    
    if os.path.exists(css_file):
        load_css(css_file)
    else:
        st.error("CSS file not found.")
 
    # Sidebar Configuration
    st.sidebar.header("Hi Isha! 👋")
    st.sidebar.write("Welcome to your Sentiment Analysis Chatbot dashboard. Customize your settings below:")
 
    # User Profile Settings
    display_name = st.sidebar.text_input("Display Name", "Isha", disabled=True)
   
    domain = st.sidebar.selectbox("Select Usecase", domains)
 
    if domain in model.domain_model:
        model_name = st.sidebar.radio("Model", model.domain_model[domain])
 
    # Language Selection
    st.sidebar.header("Language Settings")
    language = st.sidebar.selectbox("Select Language", ["English"])
   
    # Theme and Customization
    st.sidebar.header("Customization")
    theme = st.sidebar.radio("Select Theme", ["Light","dark"])
    font_size = st.sidebar.slider("Font Size", 12, 24, 16)
   
    # Feedback Mechanism
    st.sidebar.header("Feedback")
    rating = st.sidebar.slider("Rate the Chatbot", 1, 5, 3)
    feedback = st.sidebar.text_area("Provide Feedback")
   
    # Help and Documentation
    st.sidebar.header("Help & Documentation")
    if st.sidebar.button("View Help"):
        st.sidebar.write("Help documentation here...")
        
    main.page(domain)