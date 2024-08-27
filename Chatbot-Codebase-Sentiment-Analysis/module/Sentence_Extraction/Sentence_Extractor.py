import re
import spacy

nlp = spacy.load("en_core_web_sm")

conjunctions = ["but", "however", "although", "otherwise"]

pattern = r'\s*\b(?:' + '|'.join(conjunctions) + r')\b\s*'

def extract_sentences(text):
    # Split the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    sentances_conj_splitter = []
    for sentence in sentences:
      splitter = re.split(pattern, sentence)
      for sen in splitter:
        sentances_conj_splitter.append(sen)
    return sentances_conj_splitter

def extract_relevant_text(text, topics):
    # Split the text into sentences
    sentences = extract_sentences(text)

    # Dictionary to hold results
    topic_texts = {topic: ([],color) for topic,color in topics}

    # Check each sentence for topic relevance
    for sentence in sentences:
        for topic,color in topics:
            if topic.lower() in sentence.lower():
                topic_texts[topic][0].append(sentence)

    return topic_texts