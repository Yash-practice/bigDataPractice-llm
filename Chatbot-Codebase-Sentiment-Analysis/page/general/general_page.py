import streamlit as st
from models import model
from module.keywords import keywords
from module.Sentence_Extraction import Sentence_Extractor
from module.randomized_color import randomized_colors

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
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    st.markdown("""
    
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3>Welcome to the Sentiment Analysis Chatbot 🤖</h3>
    </div>
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
        </div>
    </div>
    <div style='text-align: left; margin-left: 60px;margin-bottom: 0px; padding-bottom: 0px;'>
    <h6 class='animated-heading'>🕵️ How may I assist you today?</h6>
    </div>
    """, unsafe_allow_html=True)
        
    col1,col2 = st.columns([1,12])
    with col1:
        st.write("")
    with col2:
        user_input = st.text_input("🤖 Write Some Text Here", "", key="user_input", placeholder="Type your message here...")
       
        model_name = model.domain_model[domain_name][0]
         
        if user_input and model_name:
            tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
            sentiment_mapping = model_instance.config.id2label
            sentiment = model.predict_sentiment(user_input, model_instance, tokenizer, sentiment_mapping)
            response = f"{sentiment['output']} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}%" 
            if st.session_state['Keyword_Analysis']:
                st.session_state['chat_history'] = []
                topics_keywords = keywords.keywords_extractor(user_input)
                keyword_color = "red"
                topics = [(topic,keyword_color) for topic in topics_keywords]
                highlighted_text= keywords.highlight_keywords(user_input, topics, "color")
                placeholder = st.empty()
                placeholder.markdown(highlighted_text, unsafe_allow_html=True)
                st.write(response)
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
                            # print(predict(sentence, model, tokenizer, sentiment_mapping))
            else:
                sentiment = model.predict_sentiment(user_input, model_instance, tokenizer, sentiment_mapping)
                                
                # Add user input, response, and graph to chat history
                response = f"{sentiment['output']} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}%"  # Placeholder for actual response
                st.session_state['chat_history'].append({
                    'user': user_input,
                    'response': response,
                })
 
            for chat in reversed(st.session_state['chat_history']):
                st.markdown(f'<div class="chat-response">User: {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-response">Bot: {chat["response"]}</div>', unsafe_allow_html=True)
