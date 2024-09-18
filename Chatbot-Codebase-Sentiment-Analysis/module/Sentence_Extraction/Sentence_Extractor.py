import re
import spacy

nlp = spacy.load("en_core_web_sm")

conjunctions = ["but", "however", "although", "otherwise"]

pattern = r'\s*\b(?:' + '|'.join(conjunctions) + r')\b\s*'

def split_review_text_to_rows(row, column_name):
    split_texts = extract_sentences(row[column_name])
    rows = []
    for text in split_texts:
        rows.append({
            'Reviewer_ID': row['Review_ID'],
            'Reviewer_Location': row['Reviewer_Location'],
            'Review_Text': text,
            'Year_Month': row['Year_Month'],
            'Disneyland Branch': row['Branch'],
            'Reviewer_Rating': row['Rating']
        })
    return rows

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