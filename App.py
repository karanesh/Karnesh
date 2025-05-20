import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Simulated pre-trained model (replace with actual training in production)
@st.cache_resource
def load_model():
    # For demo, we'll create a simple TF-IDF vectorizer and Logistic Regression model
    vectorizer = TfidfVectorizer(max_features=5000)
    model = LogisticRegression()
    
    # Simulated training data (replace with real dataset)
    sample_data = [
        ("This news is completely fabricated and false", 0),  # Fake
        ("Breaking news: Major event confirmed by officials", 1),  # Real
        ("Aliens landed in New York, says anonymous source", 0),  # Fake
        ("Government releases official statement on policy", 1)  # Real
    ]
    texts, labels = zip(*sample_data)
    X = vectorizer.fit_transform([preprocess_text(text) for text in texts])
    model.fit(X, labels)
    return vectorizer, model

# Load model and vectorizer
vectorizer, model = load_model()

# Streamlit app
st.title("📰 Fake News Detector")
st.write("Enter a news headline or text to check if it's fake or real.")

# User input
user_input = st.text_area("Enter news text:", height=150)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and predict
        processed_text = preprocess_text(user_input)
        X = vectorizer.transform([processed_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of being Real

        # Display result
        if prediction == 1:
            st.success(f"✅ This news is likely **Real** (Confidence: {probability:.2%})")
        else:
            st.error(f"❌ This news is likely **Fake** (Confidence: {1 - probability:.2%})")

# Instructions for users
st.markdown("""
### How it works:
1. Enter a news headline or short article text.
2. The model processes the text using NLP techniques (TF-IDF and Logistic Regression).
3. It predicts whether the news is likely **Fake** or **Real** with a confidence score.

*Note*: This is a simplified demo. For better accuracy, use a large, labeled dataset to train the model.
""")

# Run the app with: streamlit run app.py
