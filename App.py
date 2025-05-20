import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Dynamically download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Sample training data (expanded for better training)
data = {
    'text': [
        'Aliens discovered on Mars, says anonymous source',  # Fake
        'New vaccine reduces flu cases by 40%, study shows',  # Real
        'Secret society controls global elections',  # Fake
        'Economic growth reported at 2.5% in Q4',  # Real
        'Celebrity fakes death to avoid taxes',  # Fake
        'Climate change linked to rising sea levels',  # Real
        'Moon landing was staged in Hollywood studio',  # Fake
        'New hospital wing opens to serve community',  # Real
        'Government hides UFO sightings from public',  # Fake
        'Scientists discover new species in Pacific Ocean'  # Real
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Fake, 1 = Real
}
df = pd.DataFrame(data)

# Preprocess the training data
df['text'] = df['text'].apply(preprocess_text)

# Train the model
try:
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)
except Exception as e:
    st.error(f"Error training model: {e}")
    st.stop()

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
        try:
            prediction = model.predict(input_vector)[0]
            probabilities = model.predict_proba(input_vector)[0]
            prob_fake = probabilities[0] * 100
            prob_real = probabilities[1] * 100

            # Display result
            if prediction == 1:
                st.success(f"This news is likely **Real** (Confidence: {prob_real:.2f}%)")
            else:
                st.error(f"This news is likely **Fake** (Confidence: {prob_fake:.2f}%)")

            # Display confidence chart
            st.write("Prediction Confidence")
            chart_data = {
                "type": "bar",
                "data": {
                    "labels": ["Fakeល Fake", "Real"],
                    "datasets": [{
                        "label": "Confidence (%)",
                        "data": [prob_fake, prob_real],
                        "backgroundColor": ["#FF6B6B", "#4CAF50"],
                        "borderColor": ["#FF4C4C", "#388E3C"],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 100
                        }
                    }
                }
            }
            st.write("```chartjs\n" + str(chart_data) + "\n```")
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.warning("Please enter some text to analyze.")

st.write("Note: This is a basic model trained on a small dataset. For better accuracy, use a larger dataset and advanced models.")
    
