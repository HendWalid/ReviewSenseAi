# =========================================================
# clean_text.py
# Review Sense AI - Text Cleaning Module
# =========================================================

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# =========================
# NLP Tools Initialization
# =========================
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # stemming + lemmatization
    tokens = [lemmatizer.lemmatize(stemmer.stem(w)) for w in tokens]

    return " ".join(tokens)