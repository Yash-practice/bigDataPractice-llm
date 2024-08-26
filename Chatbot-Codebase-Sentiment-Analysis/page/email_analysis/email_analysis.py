import streamlit as st
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

def main(domain_name=""):
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

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

    # Email body input and analysis
    with col2:
        # Input field for email body with auto-resize based on input length
        email_body = st.text_area("✉️ Write the Email Body Here", "", key="email_body", placeholder="Type the email content here and press Enter...", height=100)

        if st.button("Analyze Email") or email_body:
            # Load sentiment analysis model
            sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

            # Perform sentiment analysis
            sentiment = sentiment_analyzer(email_body)[0]
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

            # Extract topics from the email body
            topics = extract_topics(email_body)
            total_topics = filter_stopwords(topics)

            st.write("Extracted Topics:")
            st.write(total_topics)

            # Add the email body, sentiment, and topics to chat history
            st.session_state['chat_history'].append({
                'email_body': email_body,
                'sentiment': categorized_sentiment,
                'topics': total_topics
            })

        # Display chat history
        for chat in reversed(st.session_state['chat_history']):
            st.markdown(f'<div class="chat-response">Email Body: {chat["email_body"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-response">Sentiment: {chat["sentiment"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-response">Topics: {", ".join(chat["topics"])}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
