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

def highlight_keywords(text, keywords_with_colors, color_type):
    
    if color_type=="color":
        case_keywords = get_keywords_with_case(text, [keyword for keyword,color in keywords_with_colors])
        for i in range(len(case_keywords)):
            case_keywords[i] = (case_keywords[i] , keywords_with_colors[i][1])
        keywords_with_colors = case_keywords
    
    pattern = r'(<[^>]*>)|(' + '|'.join(re.escape(keyword) for keyword, _ in keywords_with_colors) + r')'

    def replacement(match):
        if match.group(1):
            return match.group(1)  # Return the original if it's inside < >
        else:
            # Get the matched keyword
            keyword = match.group(2)
            # Find the color associated with the keyword
            for k, color in keywords_with_colors:
                if k == keyword:
                    return f'<span style="{color_type}: {color};">{keyword}</span>'
            return keyword  # Fallback (should not reach here)

    # Replace using a function to determine whether to highlight or not
    highlighted_text = re.sub(pattern, replacement, text)
    
    return highlighted_text