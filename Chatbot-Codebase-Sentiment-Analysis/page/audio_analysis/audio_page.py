import tempfile
import streamlit as st
import speech_recognition as sr
from module.keywords import keywords
from models import model
from module.randomized_color import randomized_colors
from module.Sentence_Extraction import Sentence_Extractor
import whisper
import json
import os

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, dir="temp_files") as temp_file:
        # Save the uploaded file to a temporary file
        temp_file.write(audio_file.getbuffer())
        temp_file_path = temp_file.name  
    try:
        # Perform transcription using Whisper
        result = model.transcribe(temp_file_path)
        return result['text']
    finally:
        # Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Accepting domain_name as a parameter
def main(domain_name):
    
    # Custom HTML for page layout
    st.markdown("""
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3>Speech to Text, Sentiment Analysis & Topic Extraction🤖</h3>
    </div>
    """, unsafe_allow_html=True)
     
    chat_history = []
    
    with open('chat_history.json', 'r') as file:
        chat_history = json.load(file)
        
    col1, col2 = st.columns([1, 12])

    # Empty column for spacing
    with col1:
        st.write("")

    # Chat history and input field
    with col2:
        # Load sentiment analysis model
        model_name = model.domain_model[domain_name][0]
        tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
        sentiment_mapping = model_instance.config.id2label
        # File uploader for audio files
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav","mp3","m4a","flac","ogg"])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            with st.spinner('Transcribing the file...'):
                text = transcribe_audio(uploaded_file)
            # Recognize the speech  
 
            
            st.write("Transcription:")
            placeholder = st.empty()
            placeholder.write(text)
            # Perform sentiment analysis
            sentiment = model.predict_sentiment(text, model_instance, tokenizer, sentiment_mapping)
            
            categorized_sentiment = sentiment['output']
            st.write("Sentiment Analysis:")
            response = f"{categorized_sentiment} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}"
            st.write(response)
            # Extract topics from the transcription
            topics = keywords.keywords_extractor(text)
            keyword_color = "red"
            colored_topics = [(topic,keyword_color) for topic in topics]
            highlighted_text= keywords.highlight_keywords(text, colored_topics, "color")
            placeholder.markdown(highlighted_text, unsafe_allow_html=True)
            
            col1,col2 = st.columns([3,12], vertical_alignment="center")
            with col1:
                st.write("Extracted Topics: ")
            with col2:
                chosen_topics = st.multiselect('Select Topics To Analyse', topics, label_visibility="collapsed")
                chosen_topics = [[chosen_topic,randomized_colors.get_valid_random_color()] for chosen_topic in chosen_topics]
                
            if len(chosen_topics)>0:
                relevant_texts = Sentence_Extractor.extract_relevant_text(text, chosen_topics)
                colored_sentences = []
                for topic, [sentences,color] in relevant_texts.items():
                    for sentence in sentences:
                        colored_sentences.append((sentence,color))
                highlighted_text = keywords.highlight_keywords(text, colored_sentences, "background-color")
                placeholder.markdown(highlighted_text, unsafe_allow_html=True)
                
                for topic, [sentences,color] in relevant_texts.items():
                    st.markdown(f'<h6>-------  {topic.title()}  ---------</h6>', unsafe_allow_html=True)
                    for sentence in sentences:
                        st.write(sentence)
                        sentiment = model.predict_sentiment(sentence, model_instance, tokenizer, sentiment_mapping)
                        st.write(f"Sentiment : {sentiment['output']} sentiment with score of {sentiment['probs'][sentiment['output']]*100:.2f}%")
            
            json_chat = {
                "data_type": "Audio",
                "analysis_type" : f"{domain_name}",
                "value": f"{text}",
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
            if chat["data_type"]=="Audio":
                st.markdown(f"**Transcription:** {chat['value']}")
                st.markdown(f"**Sentiment:** {chat['response']}")
                commaseptopics = ",".join(chat['keywords'])
                st.markdown(f"**Topics:** {commaseptopics}")
                st.markdown('<hr>', unsafe_allow_html=True)

if __name__ == "__main__":
    main("")
