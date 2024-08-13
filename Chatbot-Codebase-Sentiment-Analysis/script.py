from constants import domain_constant,model_constant
from models import model
from graphs import graph
import streamlit as st
import os

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

domain_model = { domain_constant.GENERAL : [model_constant.TWITTER_ROBERTA_BASE_SENTIMENT_LATEST] ,
                 domain_constant.SOCIAL_MEDIA: [model_constant.ROBERTA_BASE_GO_EMOTIONS]}

domains = [domain_constant.GENERAL, domain_constant.SOCIAL_MEDIA, domain_constant.CUSTOMER_REVIEW]

if __name__ == "__main__":
        
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

    if domain in domain_model:
        model_name = st.sidebar.radio("Model", domain_model[domain])
    
    # Language Selection
    st.sidebar.header("Language Settings")
    language = st.sidebar.selectbox("Select Language", ["English"])
    
    # Theme and Customization
    st.sidebar.header("Customization")
    theme = st.sidebar.radio("Select Theme", ["Light"])
    font_size = st.sidebar.slider("Font Size", 12, 24, 16)
    
    # Feedback Mechanism
    st.sidebar.header("Feedback")
    rating = st.sidebar.slider("Rate the Chatbot", 1, 5, 3)
    feedback = st.sidebar.text_area("Provide Feedback")
    
    # Help and Documentation
    st.sidebar.header("Help & Documentation")
    if st.sidebar.button("View Help"):
        st.sidebar.write("Help documentation here...")
    
    # Main Interface for the Chatbot with Animated Header and Emoji Slider
    st.markdown("""
        <div class="slider-container">
        <div class="slider">
            <div>😇</div>
            <div>😠</div>
            <div>😢</div>
            <div>😊</div>
            <div>😃</div>
            <div>😎</div>
            <div>😴</div>
            <div>😮</div>
            <div>🙁</div>
            <div>😇</div>
            <div>😛</div>
            <div>😠</div> <!-- Repeat emojis to create a seamless loop -->
            <div>😢</div>
            <div>😊</div>
            <div>😃</div>
            <div>😎</div>
            <div>😴</div>
            <div>😮</div>
            <div>🙁</div>
            <div>😇</div>
            <div>😛</div>
        </div>
    </div>
    <div style='text-align: left; margin-left: 60px;'>
        <h3>Welcome to the Sentiment Analysis Chatbot 🤖</h3>
    </div>
    <div style='text-align: left; margin-left: 60px;'>
    <h6 class='animated-heading'>🕵️ How may I assist you today?</h6>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 12])
 
    # Empty column for spacing
    with col1:
        st.write("")
 
    # Input field aligned at the bottom with a polished container
    with col2:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            user_input = st.text_input("Write Some Text Here", "", key="user_input", placeholder="Type your message here...")
            st.markdown('</div>', unsafe_allow_html=True)
 
    # Displaying the response with improved styling
    if user_input and model_name:
        tokenizer,model_instance = model.load_roberta_model(f'models/{model_name}/model',f'models/{model_name}/tokenizer')
        sentiment_mapping = model_instance.config.id2label
        sentiment = model.predict_sentiment(user_input,model_instance,tokenizer,sentiment_mapping)
        st.write(sentiment)
        st.markdown(f'<div class="chat-response">You said: {user_input}</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-response">Chatbot response goes here...</div>', unsafe_allow_html=True)

        zones = [{
            'name':'Negative',
            'color':'red',
            'range': [0,(sentiment['probs']['negative'])*100]
        },
                 {
            'name':'Neutral',
            'color':'yellow',
            'range': [(sentiment['probs']['negative'])*100,(sentiment['probs']['negative']+sentiment['probs']['neutral'])*100]
        },{
            'name':'Positive',
            'color':'green',
            'range': [(sentiment['probs']['negative']+sentiment['probs']['neutral'])*100,100]
        }]
        
        # Plot gauge
        gauge_fig = graph.plot_gauge(zones)
        st.plotly_chart(gauge_fig)
