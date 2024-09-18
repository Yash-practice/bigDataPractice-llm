import tempfile
import streamlit as st
from module.keywords import keywords
from models import model
from module.randomized_color import randomized_colors
from module.Sentence_Extraction import Sentence_Extractor
import whisper
import pandas as pd
import json
import os

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, dir="temp_files") as temp_file:
        temp_file.write(audio_file.getbuffer())
        temp_file_path = temp_file.name

        result = model.transcribe(temp_file_path)
        return result['text']

def main(domain_name):
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'selected_file_index' not in st.session_state:
        st.session_state['selected_file_index'] = None

    st.markdown("""
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3>Speech to Text, Sentiment Analysis & Topic Extraction🤖</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 12])
    
    chat_history = []
    
    with open('chat_history.json', 'r') as file:
        chat_history = json.load(file)

    with col1:
        st.write("")

    with col2:
        model_name = model.domain_model[domain_name][0]
        tokenizer, model_instance = model.load_roberta_model(f'{model_name}/model', f'{model_name}/tokenizer')
        sentiment_mapping = model_instance.config.id2label

        uploaded_files = st.file_uploader("Choose audio files", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=True)

        if uploaded_files:
            st.write("Uploaded Audio Files:")

            # Create a dropdown menu to select a file
            file_names = [file.name for file in uploaded_files]
            selected_file_name = st.selectbox("Select a file to analyze", ["None"] + file_names)

            if selected_file_name != "None":
                selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
                st.write(f"### Selected File: {selected_file.name}")
                st.audio(selected_file)

                with st.spinner('Transcribing the file...'):
                    text = transcribe_audio(selected_file)

                st.write("**Transcription:**")
                placeholder = st.empty()
                placeholder.write(text)

                sentiment = model.predict_sentiment(text, model_instance, tokenizer, sentiment_mapping)
                categorized_sentiment = sentiment['output']
                st.write(f"**{model.domain_model[domain_name][1]} Analysis:**")
                st.write(f"{model.domain_model[domain_name][1]}: {categorized_sentiment}")
                st.write(f"Score: {sentiment['probs'][sentiment['output']]*100:.2f}%")

                topics = keywords.keywords_extractor(text)
                keyword_color = "red"
                colored_topics = [(topic, keyword_color) for topic in topics]
                highlighted_text = keywords.highlight_keywords(text, colored_topics, "color")
                placeholder.markdown(highlighted_text, unsafe_allow_html=True)

                col1, col2 = st.columns([3, 12])
                with col1:
                    st.write("**Extracted Topics:**")
                with col2:
                    chosen_topics = st.multiselect('Select Topics To Analyse', topics, label_visibility="collapsed")
                    chosen_topics = [[chosen_topic, randomized_colors.get_valid_random_color()] for chosen_topic in chosen_topics]

                if chosen_topics:
                    relevant_texts = Sentence_Extractor.extract_relevant_text(text, chosen_topics)
                    colored_sentences = []
                    for topic, [sentences, color] in relevant_texts.items():
                        for sentence in sentences:
                            colored_sentences.append((sentence, color))
                    highlighted_text = keywords.highlight_keywords(text, colored_sentences, "background-color")
                    placeholder.markdown(highlighted_text, unsafe_allow_html=True)

                    for topic, [sentences, color] in relevant_texts.items():
                        st.markdown(f'<h6>-------  {topic.title()}  ---------</h6>', unsafe_allow_html=True)
                        for sentence in sentences:
                            st.write(sentence)
                            sentiment = model.predict_sentiment(sentence, model_instance, tokenizer, sentiment_mapping)
                            st.write(f"**{model.domain_model[domain_name][1]}:** {sentiment['output']} {model.domain_model[domain_name][1].lower()} with score of {sentiment['probs'][sentiment['output']]*100:.2f}%")

                json_chat = {
                "data_type": "Audio",
                "analysis_type" : f"{domain_name}",
                "value": f"{text}",
                "keywords": topics,
                "response": categorized_sentiment
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

            # Store results for all files in a list
            results = []

            for file in uploaded_files:
                with st.spinner(f'Processing {file.name}...'):
                    # Transcribe the audio
                    transcription = transcribe_audio(file)

                    # Perform sentiment analysis on the transcription
                    sentiment = model.predict_sentiment(transcription, model_instance, tokenizer, sentiment_mapping)
                    categorized_sentiment = sentiment['output']
                    sentiment_score = sentiment['probs'][categorized_sentiment] * 100

                    # Extract topics from the transcription
                    topics = keywords.keywords_extractor(transcription)
                    extracted_topics = ", ".join(topics)

                    # Store results in a dictionary for the DataFrame
                    results.append({
                        'File Name': file.name,
                        # 'Transcription': transcription,
                        'Sentiment': categorized_sentiment,
                        'Sentiment Score': f"{sentiment_score:.2f}%",
                        # 'Topics': extracted_topics
                    })

            # Create a DataFrame for the results
            results_df = pd.DataFrame(results)

            # Create a column layout for the button and the table
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write("")  # Placeholder for alignment

            with col2:
                if st.button("Show Sentiment Analysis Table"):
                    st.write("### Sentiment Analysis Table")
                    st.markdown(
                        """
                        <style>
                        .dataframe {
                            width: 100%;
                            max-width: 100%;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.dataframe(results_df, use_container_width=True)

        st.markdown(f'<h6>--------------------------------------------------------  Chat History  ---------------------------------------------------------</h6>', unsafe_allow_html=True)
        for chat in reversed(chat_history):
            if chat["data_type"]=="Audio":
                st.markdown(f"**Transcription:** {chat['value']}")
                st.markdown(f"**Output:** {chat['response']}")
                commaseptopics = ",".join(chat['keywords'])
                st.markdown(f"**Topics:** {commaseptopics}")
                st.markdown('<hr>', unsafe_allow_html=True)

if __name__ == "__main__":
    main("your_domain_name_here")
