import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Sample training data (replace with a larger dataset for better accuracy)
data = {
    'text': [
        'The government is hiding alien evidence',  # Fake
        'New study confirms climate change impact',  # Real
        'Celebrity caught in scandal with no proof',  # Fake
        'Local hospital opens new wing for patients',  # Real
        'Secret society controls global economy',  # Fake
        'Economy grows by 2% in Q3',  # Real
    ],
    'label': [0, 1, 0, 1, 0, 1]  # 0 = Fake, 1 = Real
}
df = pd.DataFrame(data)

# Preprocess the training data
df['text'] = df['text'].apply(preprocess_text)

# Train a simple model
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']
model = LogisticRegression()
model.fit(X, y)

# Streamlit app
st.title("Fake News Detector")
st.write("Enter a news headline or article snippet to check if it's real or fake.")

# User input
user_input = st.text_area("Enter news text:", height=150)

if st.button("Check News"):
    if user_input:
        # Preprocess user input
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input]).toarray()
        
        # Predict
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][prediction] * 100

        # Display result
        if prediction == 1:
            st.success(f"This news is likely **Real** (Confidence: {probability:.2f}%)")
        else:
            st.error(f"This news is likely **Fake** (Confidence: {probability:.2f}%)")
    else:
        st.warning("Please enter some text to analyze.")

st.write("Note: This is a basic model trained on a small dataset. For better accuracy, use a larger dataset and advanced models.")
