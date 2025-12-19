import streamlit as st
import pandas as pd
import joblib
import re
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


NEWS_API_KEY = "Your News API Key"     #Place your NewsAPI Key here


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# Load or train model
@st.cache_resource
def load_model():
    if os.path.exists("vectorizer.pkl") and os.path.exists("fake_news_model.pkl"):
        st.info("🔁 Loading existing model and vectorizer...")
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("fake_news_model.pkl")
    else:
        st.info("🔧 Training model from scratch...")

        df = pd.read_csv("True.csv")  # CSV must have 'text' and 'label'
        df["text"] = df["text"].apply(clean_text)

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]

        model = LogisticRegression()
        model.fit(X, y)

        joblib.dump(vectorizer, "vectorizer.pkl")
        joblib.dump(model, "fake_news_model.pkl")
        st.success("✅ Model trained and saved.")

    return vectorizer, model

# Load model and vectorizer
vectorizer, model = load_model()

# Fetch real-time news from NewsAPI
def fetch_live_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", [])
        return [(a["title"], a.get("description", "")) for a in articles]
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

# UI
st.title("📰 Real-Time Fake News Detector")
st.markdown("Check **your own news input** or scan **live headlines** from the web.")

# Manual check
user_input = st.text_area("✏️ Type or paste a news headline/article:")

if st.button("🔍 Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader(f"Prediction: {label}")

# Live News Detection
st.markdown("---")
if st.button("🔄 Detect Fake News from Live Headlines"):
    st.info("⏳ Fetching latest headlines...")
    headlines = fetch_live_news()
    if headlines:
        st.success(f"✅ Fetched {len(headlines)} headlines.")
        for i, (title, desc) in enumerate(headlines, start=1):
            full_text = clean_text(title + " " + desc)
            vectorized = vectorizer.transform([full_text])
            prediction = model.predict(vectorized)[0]
            label = "🟢 Real" if prediction == 1 else "🔴 Fake"
            st.markdown(f"**{i}. {title}**\n\n> Prediction: {label}\n")

