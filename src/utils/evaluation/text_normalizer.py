# import re

# STOPWORDS = set([
#     "a", "an", "the", "and", "or", "but", "if", "then", "because", "as", "of", "at", "by",
#     "for", "with", "about", "against", "between", "into", "through", "during", "before",
#     "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
#     "over", "under", "again", "further", "once", "here", "there", "when", "where",
#     "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
#     "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
# ])

# def normalize_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     tokens = text.split()

#     # Remove stopwords
#     tokens = [t for t in tokens if t not in STOPWORDS]

#     # Synonym replacements (can expand)
#     synonyms = {
#         'lever': 'stalk',
#         'button side down': 'button-side down',
#         'autopilot settings': 'controls autopilot settings',
#         'autosteer': 'autopilot',
#         'press': 'push',
#         'hold the cover': 'align the cover',
#         'battery': 'cr2032 battery',
#         'remove the bottom cover': 'release the bottom cover'
#     }
#     # Replace phrases in tokens
#     text_norm = ' '.join(tokens)
#     for k, v in synonyms.items():
#         text_norm = text_norm.replace(k, v)

#     # Normalize whitespace again
#     return ' '.join(text_norm.split())
import re
import string

def normalize_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Optionally remove articles ('a', 'an', 'the')
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
