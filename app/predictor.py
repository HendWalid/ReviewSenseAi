# =========================================================
# predictor.py
# ReviewSense AI - Prediction Logic
# =========================================================

import joblib
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.preprocessing.clean_text import clean_text

def load_models():
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    model = joblib.load('../models/logistic_regression.pkl')
    return vectorizer, model

def predict_sentiment(review, vectorizer, model):
    cleaned = clean_text(review)
    transformed = vectorizer.transform([cleaned])
    label = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed)[0][label]
    return label, confidence