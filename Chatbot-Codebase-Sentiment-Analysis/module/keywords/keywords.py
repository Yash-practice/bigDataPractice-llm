import nltk
import re
import string
from rake_nltk import Rake

nltk.download('stopwords')
nltk.download('punkt_tab')

def is_special(characters):
  special_chars = set(string.punctuation)
  special_chars.add('’')
  return all(char in special_chars for char in characters)

def get_keywords_with_case(text, keywords):
    case_map = []
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            case_map.append(match.group())
    return case_map

def keywords_extractor(text):
    rake = Rake()

    rake.extract_keywords_from_text(text)
    keywords = set(rake.get_ranked_phrases())
    keywords = [string for string in keywords if not is_special(string)]
    
    return keywords

def highlight_keywords(text, keywords, color_type):
    for keyword, color in keywords:
        if keyword in text:
            keyword = keyword.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace(keyword, f'<span style="{color_type}: {color};">{keyword}</span>')
    return text