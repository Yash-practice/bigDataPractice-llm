import streamlit as st
from models import model
from module.keywords import keywords
from module.Sentence_Extraction import Sentence_Extractor
from module.randomized_color import randomized_colors
import json

def onChangeAllKeywords():
    if st.session_state['All_Keywords']:
        st.session_state['Topics_Keywords'] = True
        st.session_state['Sentimental_Keywords'] = True

def general_search(domain_name):
    
    st.markdown("""
    <script>
    function handleClick(keyword) {
        window.parent.postMessage({ keyword: keyword }, "*");
    }
    </script>
    """, unsafe_allow_html=True)
       # Initialize chat history in session state
    
    if 'function_call' not in st.session_state:
        st.session_state['function_call'] = None   
    
    st.markdown("""
    
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3>Welcome to the Sentiment Analysis Chatbot 🤖</h3>
    </div>
    <div style='text-align: left; margin-left: 60px;margin-bottom: 0px; padding-bottom: 0px;'>
    <h6 class='animated-heading'>🕵️ How may I assist you today?</h6>
    </div>
    """, unsafe_allow_html=True)
    
    chat_history = []
    
    with open('chat_history.json', 'r') as file:
        chat_history = json.load(file)
    
    col1,col2 = st.columns([1,12])
    with col1:
        st.write("")
    with col2:
        user_input = st.text_input("🤖 Write Some Text Here", "", key="user_input", placeholder="Type your message here...")
       
        if st.button("Analyze") or user_input:
            model_name = model.domain_model[domain_name][0]
            
            if user_input and model_name:
                tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
                sentiment_mapping = model_instance.config.id2label
                sentiment = model.predict_sentiment(user_input, model_instance, tokenizer, sentiment_mapping)
                response = f"{sentiment['output']} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}%" 
                topics_keywords = []
                if st.session_state['Keyword_Analysis']:
                    topics_keywords = keywords.keywords_extractor(user_input)
                    keyword_color = "red"
                    topics = [(topic,keyword_color) for topic in topics_keywords]
                    highlighted_text= keywords.highlight_keywords(user_input, topics, "color")
                    placeholder = st.empty()
                    placeholder.markdown(highlighted_text, unsafe_allow_html=True)
                    chosen_topics = st.multiselect('Select Topics To Analyse', topics_keywords)
                    chosen_topics = [[chosen_topic,randomized_colors.get_valid_random_color()] for chosen_topic in chosen_topics]
                    if len(chosen_topics)>0:
                        relevant_texts = Sentence_Extractor.extract_relevant_text(user_input, chosen_topics)
                        colored_sentences = []
                        for topic, [sentences,color] in relevant_texts.items():
                            for sentence in sentences:
                                colored_sentences.append((sentence,color))
                        highlighted_text = keywords.highlight_keywords(user_input, colored_sentences, "background-color")
                        placeholder.markdown(highlighted_text, unsafe_allow_html=True)
                        for topic, [sentences,color] in relevant_texts.items():
                            st.markdown(f'<h6>-------  {topic.title()}  ---------</h6>', unsafe_allow_html=True)
                            for sentence in sentences:
                                st.write(sentence)
                                sentiment = model.predict_sentiment(sentence, model_instance, tokenizer, sentiment_mapping)
                                st.write(f"Sentiment : {sentiment['output']} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}%")
                                
                else:
                    sentiment = model.predict_sentiment(user_input, model_instance, tokenizer, sentiment_mapping)               
                    response = f"{sentiment['output']} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}%"  # Placeholder for actual response
                    st.write(user_input)
                    st.markdown(f'<h6>-------  Sentiment  ---------</h6>', unsafe_allow_html=True)
                    st.write(response)
                    
                json_chat = {
                    "data_type": "Text",
                    "analysis_type" : f"{domain_name}",
                    "value": f"{user_input}",
                    "keywords": topics_keywords,
                    "response": response
                }
                
                if 'json_chat' in st.session_state:
                    if json_chat!=st.session_state['json_chat']:
                        chat_history.append(json_chat)          
                        with open('chat_history.json', 'w') as file:
                            json.dump(chat_history, file)
                else:
                    chat_history.append(json_chat)          
                    with open('chat_history.json', 'w') as file:
                        json.dump(chat_history, file)
                st.session_state['json_chat'] = json_chat
                
        st.markdown(f'<h6>--------------------------------------------------------  Chat History  ---------------------------------------------------------</h6>', unsafe_allow_html=True)
        for chat in reversed(chat_history):
            if chat["data_type"]=="Text":
                st.markdown(f'<div class="chat-response">User: {chat["value"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-response">Bot: {chat["response"]}</div>', unsafe_allow_html=True)
                if len(chat["keywords"])>0:
                    st.markdown(f'<div class="chat-response">keywords: {",".join(chat["keywords"])}</div>', unsafe_allow_html=True)
                st.markdown('<hr>', unsafe_allow_html=True)