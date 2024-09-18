import streamlit as st
from models import model
from module.keywords import keywords
from module.randomized_color import randomized_colors
from module.Sentence_Extraction import Sentence_Extractor
import json

def main(domain_name=""):
    # Initialize chat history in session state

    # Custom HTML for page layout
    st.markdown("""
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3>Email Sentiment Analysis & Topic Extraction 🤖</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 12])

    # Empty column for spacing
    with col1:
        st.write("")

    chat_history = []
    
    with open('chat_history.json', 'r') as file:
        chat_history = json.load(file)
        
    # Email body input and analysis
    with col2:
        # Input field for email body with auto-resize based on input length
        email_body = st.text_area("✉️ Write the Email Body Here", "", key="email_body", placeholder="Type the email content here and press Enter...", height=100)

        if st.button("Analyze Email") or email_body:
            
            st.write("Email Body: ")
            placeholder = st.empty()
            placeholder.write(email_body)
            
            # Load sentiment analysis model
            model_name = model.domain_model[domain_name][0]
            tokenizer, model_instance = model.load_roberta_model(f'{model_name}/model', f'{model_name}/tokenizer')
            sentiment_mapping = model_instance.config.id2label
            
            # Perform sentiment analysis
            sentiment = model.predict_sentiment(email_body, model_instance, tokenizer, sentiment_mapping)
            
            categorized_sentiment = sentiment['output']
            response = f"{categorized_sentiment} {model.domain_model[domain_name][1].lower()} with score of {sentiment['probs'][sentiment['output']]*100:.2f}%"
            st.write(f"{model.domain_model[domain_name][1]} Analysis:")
            st.write(response)

            # Extract topics from the email body
            topics = keywords.keywords_extractor(email_body)
            keyword_color = "red"
            colored_topics = [(topic,keyword_color) for topic in topics]
            highlighted_text= keywords.highlight_keywords(email_body, colored_topics, "color")
            placeholder.markdown(highlighted_text, unsafe_allow_html=True)
            
            col1,col2 = st.columns([3,12], vertical_alignment="center")
            with col1:
                st.write("Extracted Topics: ")
            with col2:
                chosen_topics = st.multiselect('Select Topics To Analyse', topics, label_visibility="collapsed")
                chosen_topics = [[chosen_topic,randomized_colors.get_valid_random_color()] for chosen_topic in chosen_topics]

            if len(chosen_topics)>0:
                relevant_texts = Sentence_Extractor.extract_relevant_text(email_body, chosen_topics)
                colored_sentences = []
                for topic, [sentences,color] in relevant_texts.items():
                    for sentence in sentences:
                        colored_sentences.append((sentence,color))
                highlighted_text = keywords.highlight_keywords(email_body, colored_sentences, "background-color")
                placeholder.markdown(highlighted_text, unsafe_allow_html=True)
                    
                for topic, [sentences,color] in relevant_texts.items():
                    st.markdown(f'<h6>-------  {topic.title()}  ---------</h6>', unsafe_allow_html=True)
                    for sentence in sentences:
                        st.write(sentence)
                        sentiment = model.predict_sentiment(sentence, model_instance, tokenizer, sentiment_mapping)
                        st.write(f"{model.domain_model[domain_name][1]} : {sentiment['output']} {model.domain_model[domain_name][1].lower()} with score of {sentiment['probs'][sentiment['output']]*100:.2f}%")
        
            json_chat = {
                "data_type": "Email",
                "analysis_type" : f"{domain_name}",
                "value": f"{email_body}",
                "keywords": topics,
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
            if chat["data_type"]=="Email":
                st.markdown(f'<div class="chat-response">User: {chat["value"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-response">Bot: {chat["response"]}</div>', unsafe_allow_html=True)
                if len(chat["keywords"])>0:
                    st.markdown(f'<div class="chat-response">keywords: {",".join(chat["keywords"])}</div>', unsafe_allow_html=True)
                st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
