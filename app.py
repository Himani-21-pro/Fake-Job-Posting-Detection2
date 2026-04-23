import streamlit as st
import pickle
import re

# Load saved model and tfidf
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Text cleaning function (same as used in training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# UI Design
st.title("💼 Fake Job Posting Detection System")
st.write("Enter a job description to check whether it is Fake or Real.")

# Input box
user_input = st.text_area("Enter Job Description")

# Prediction button
if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)

        if result[0] == 1:
            st.error("🚨 This is a Fake Job Posting")
        else:
            st.success("✅ This is a Real Job Posting")
    else:
        st.warning("Please enter some text")