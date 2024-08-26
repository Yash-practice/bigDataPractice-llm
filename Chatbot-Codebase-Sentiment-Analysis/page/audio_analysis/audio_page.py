import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_topics(text):
    sentences = sent_tokenize(text)
    noun_phrases = []

    for sentence in sentences:
        doc = nlp(sentence)
        for chunk in doc.noun_chunks:
            noun_phrases.append(chunk.text)
        for ent in doc.ents:
            noun_phrases.append(ent.text)

    tagged_words = pos_tag(noun_phrases)
    nouns = [word for word, tag in tagged_words if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
    return nouns

def filter_stopwords(topics):
    stop_words = set(stopwords.words('english'))
    topics_without_stopwords = []

    for topic in set(topics):  # Using set to remove duplicates
        words = word_tokenize(topic)
        filtered_topic = " ".join([word for word in words if word.lower() not in stop_words])
        if filtered_topic:  # Add non-empty phrases only
            topics_without_stopwords.append(filtered_topic)

    return set(topics_without_stopwords)

# Accepting domain_name as a parameter
def main(domain_name):
    # Initialize chat history in session state
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
        sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

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
                st.write(text)

                # Perform sentiment analysis
                sentiment = sentiment_analyzer(text)[0]
                sentiment_label = sentiment['label']
                sentiment_score = sentiment['score']

                # Map sentiment to categories
                sentiment_categories = {
                    'LABEL_0': 'Negative',
                    'LABEL_1': 'Neutral',
                    'LABEL_2': 'Positive'
                }
                categorized_sentiment = sentiment_categories.get(sentiment_label, 'Unknown')

                st.write("Sentiment Analysis:")
                st.write(f"Sentiment: {categorized_sentiment}")
                # st.write(f"Score: {sentiment_score:.2f}")

                # Extract topics from the transcription
                topics = extract_topics(text)
                total_topics = filter_stopwords(topics)

                st.write("Extracted Topics:")
                st.write(total_topics)

                # Store the transcription, sentiment, and topics in session state
                st.session_state['chat_history'].append({
                    'transcription': text,
                    'sentiment': categorized_sentiment,
                    'topics': total_topics
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
            st.markdown(f"**Topics:** {', '.join(entry['topics'])}")
            st.markdown('<hr>', unsafe_allow_html=True)

if __name__ == "__main__":
    main("")
