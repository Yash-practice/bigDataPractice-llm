import streamlit as st
import speech_recognition as sr
from module.keywords import keywords
from models import model
from module.randomized_color import randomized_colors
from module.Sentence_Extraction import Sentence_Extractor

# Accepting domain_name as a parameter
def main(domain_name):
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Custom HTML for page layout
    st.markdown("""
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3>Speech to Text, Sentiment Analysis & Topic Extraction🤖</h3>
    </div>
    """, unsafe_allow_html=True)
     
    col1, col2 = st.columns([1, 12])

    # Empty column for spacing
    with col1:
        st.write("")

    # Chat history and input field
    with col2:
        # Load sentiment analysis model
        model_name = model.domain_model[domain_name][0]
        tokenizer, model_instance = model.load_roberta_model(f'{model_name}/model', f'{model_name}/tokenizer')
        sentiment_mapping = model_instance.config.id2label
        # File uploader for audio files
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')

            # Convert uploaded file to speech
            recognizer = sr.Recognizer()
            audio_data = sr.AudioFile(uploaded_file)

            with audio_data as source:
                audio = recognizer.record(source)
            
            # Recognize the speech
            try:
                text = recognizer.recognize_google(audio)
                st.write("Transcription:")
                placeholder = st.empty()
                placeholder.write(text)

                # Perform sentiment analysis
                sentiment = model.predict_sentiment(text, model_instance, tokenizer, sentiment_mapping)
                
                categorized_sentiment = sentiment['output']

                st.write("Sentiment Analysis:")
                st.write(f"Sentiment: {categorized_sentiment}")
                st.write(f"Score: {sentiment['probs'][sentiment['output']]*100:.2f}")

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

                # Store the transcription, sentiment, and topics in session state
                st.session_state['chat_history'].append({
                    'transcription': text,
                    'sentiment': categorized_sentiment,
                    'topics': ", ".join(topics)
                })

            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")

    # Display chat history
    if st.session_state['chat_history']:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("<h4>Chat History:</h4>", unsafe_allow_html=True)
        for entry in st.session_state['chat_history']:
            st.markdown(f"**Transcription:** {entry['transcription']}")
            st.markdown(f"**Sentiment:** {entry['sentiment']}")
            st.markdown(f"**Topics:** {entry['topics']}")
            st.markdown('<hr>', unsafe_allow_html=True)

if __name__ == "__main__":
    main("")
