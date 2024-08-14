from constants import domain_constant, model_constant
from models import model
from graphs import graph
import streamlit as st
import os
 
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
domain_model = {
    domain_constant.GENERAL: [model_constant.TWITTER_ROBERTA_BASE_SENTIMENT_LATEST],
    domain_constant.SOCIAL_MEDIA: [model_constant.ROBERTA_BASE_GO_EMOTIONS]
}
 
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
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
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
 
    # Chat history and input field
    with col2:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
           
            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
 
            # Display existing chat history
            for chat in st.session_state['chat_history']:
                st.markdown(f'<div class="chat-response">User: {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-response">Bot: {chat["response"]}</div>', unsafe_allow_html=True)
                if chat.get("graph"):
                    st.plotly_chart(chat["graph"], use_container_width=True)
 
            st.markdown('</div>', unsafe_allow_html=True)
       
        # Input field fixed at the bottom
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        user_input = st.text_input("🤖 Write Some Text Here", "", key="user_input", placeholder="Type your message here...")
        st.markdown('</div>', unsafe_allow_html=True)
 
        # Handle user input and update chat history
        if user_input and model_name:
            tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
            sentiment_mapping = model_instance.config.id2label
            sentiment = model.predict_sentiment(user_input, model_instance, tokenizer, sentiment_mapping)
 
            zones = [{
                'name': 'Negative',
                'color': 'red',
                'range': [0, (sentiment['probs']['negative']) * 100]
            },
            {
                'name': 'Neutral',
                'color': 'yellow',
                'range': [(sentiment['probs']['negative']) * 100, (sentiment['probs']['negative'] + sentiment['probs']['neutral']) * 100]
            },
            {
                'name': 'Positive',
                'color': 'green',
                'range': [(sentiment['probs']['negative'] + sentiment['probs']['neutral']) * 100, 100]
            }]
 
            # Plot gauge
            gauge_fig = graph.plot_gauge(zones)
 
            # Add user input, response, and graph to chat history
            response = f"Response to '{user_input}'"  # Placeholder for actual response
            st.session_state['chat_history'].append({
                'user': user_input,
                'response': response,
                'graph': gauge_fig
            })
 
            # Display new chat entry
            st.markdown(f'<div class="chat-response">User: {user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-response">Bot: {response}</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_fig, use_container_width=True)