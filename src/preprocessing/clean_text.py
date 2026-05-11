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
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# =========================
# NLP Tools Initialization
# =========================
custom_stopwords = {
    'film', 'movie', 'movies', 'films',
    'watch', 'watching', 'watched',
    'show', 'series', 'episode',
    'good', 'bad', 'great', 'really',
    'make', 'made', 'way', 'even',
    'one', 'two', 'first', 'second',
    'see', 'seen', 'time', 'story',
    'think', 'know', 'get', 'much'
}
stop_words = set(stopwords.words('english')).union(custom_stopwords)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove html tags
    text = re.sub(r'<.*?>', '', text)

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

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

    # remove short words
    tokens = [w for w in tokens if len(w) > 2]

    # lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)