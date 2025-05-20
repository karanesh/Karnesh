import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Streamlit app
st.title("📰 Fake News Detection")
st.write("Enter a news headline or article text to check if it's fake or real.")

# Sample dataset (for demonstration; in practice, use a larger dataset)
data = {
    'text': [
        "Scientists confirm moon is made of cheese",
        "New study reveals benefits of regular exercise",
        "Aliens invade New York, officials say",
        "Government passes new healthcare reform bill",
        "Elvis Presley found alive in Texas",
        "Local team wins national championship"
    ],
    'label': [0, 1, 0, 1, 0, 1]  # 0 = Fake, 1 = Real
}
df = pd.DataFrame(data)

# Clean the text data
df['text'] = df['text'].apply(clean_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X, y)

# User input
user_input = st.text_area("Enter news headline or article text:", height=150)

if st.button("Check News"):
    if user_input:
        # Clean and vectorize user input
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        
        # Predict
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][prediction] * 100
        
        # Display result
        if prediction == 1:
            st.success(f"This news is likely **Real** ({probability:.2f}% confidence).")
        else:
            st.error(f"This news is likely **Fake** ({probability:.2f}% confidence).")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.write("Note: This is a simple demo using a small dataset. For better accuracy, use a larger, well-curated dataset and advanced NLP models.")
