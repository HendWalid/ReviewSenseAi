# =========================================================
# app.py
# ReviewSense AI - Streamlit Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from predictor import load_models, predict_sentimentc

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="🎬",
    layout="centered"
)

# =========================
# Load Models
# =========================
@st.cache_resource
def get_models():
    return load_models()

vectorizer, model = get_models()

# =========================
# Header
# =========================
st.title("🎬 ReviewSense AI")
st.write("Analyze movie reviews instantly. Paste any review below and find out if it's positive or negative.")
st.divider()

# =========================
# User Input
# =========================
review = st.text_area(
    "Paste your movie review here",
    height=200,
    placeholder="e.g. This movie was absolutely fantastic! The acting was superb..."
)

# =========================
# Prediction
# =========================
if st.button("Analyze Review", use_container_width=True):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing your review..."):
            label, confidence = predict_sentiment(review, vectorizer, model)

        st.divider()

        if label == 1:
            st.success(f"✅ Positive Review — {confidence:.1%} confidence")
        else:
            st.error(f"❌ Negative Review — {confidence:.1%} confidence")

        # =========================
        # Word Importance
        # =========================
        st.divider()
        st.write("### 🔍 Most Influential Words")

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        word_importance = pd.DataFrame({
            'word': feature_names,
            'importance': coefficients
        })

        if label == 1:
            top_words = word_importance.nlargest(10, 'importance')
            st.write("Words that pushed toward **positive:**")
        else:
            top_words = word_importance.nsmallest(10, 'importance')
            st.write("Words that pushed toward **negative:**")

        st.bar_chart(top_words.set_index('word')['importance'])

# =========================
# Footer
# =========================
st.divider()
st.caption("ReviewSense AI — Built with Streamlit, Scikit-learn & NLP")