import streamlit as st
from models import model
from graphs import graph
import os

def general_search(domain_name):
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
 
        model_name = model.domain_model[domain_name][0]
        
        # Handle user input and update chat history
        if user_input and model_name:
            tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
            sentiment_mapping = model_instance.config.id2label
            sentiment = model.predict_sentiment(user_input, model_instance, tokenizer, sentiment_mapping)
 
            # zones = [{
            #     'name': 'Negative',
            #     'color': 'red',
            #     'range': [0, (sentiment['probs']['negative']) * 100]
            # },
            # {
            #     'name': 'Neutral',
            #     'color': 'yellow',
            #     'range': [(sentiment['probs']['negative']) * 100, (sentiment['probs']['negative'] + sentiment['probs']['neutral']) * 100]
            # },
            # {
            #     'name': 'Positive',
            #     'color': 'green',
            #     'range': [(sentiment['probs']['negative'] + sentiment['probs']['neutral']) * 100, 100]
            # }]
            
            labels = list(sentiment["probs"].keys())
            scores = [value *100 for value in list(sentiment["probs"].values())]
            
            
            fig = graph.barChart(labels,scores,"Analysis", "Sentiment", "score")
             
            # Add user input, response, and graph to chat history
            response = f"Response to '{user_input}'"  # Placeholder for actual response
            st.session_state['chat_history'].append({
                'user': user_input,
                'response': response,
                'graph': fig
            })
 
            # # Display new chat entry
            st.markdown(f'<div class="chat-response">User: {user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-response">Bot: {response}</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)