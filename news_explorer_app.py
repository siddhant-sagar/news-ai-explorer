import streamlit as st
import requests
import os
import pandas as pd
from textblob import TextBlob
import google.generativeai as genai
import nltk

# Download necessary NLTK resources (needed for Streamlit Cloud)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Gemini API key from env or Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-2.0-flash")

# Function to summarize a news snippet
def summarize_text_gemini(text, topic):
    prompt = f"Summarize the following news article about '{topic}' in 2-3 sentences:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Get sentiment label
def get_sentiment(summary):
    sentiment_score = TextBlob(summary).sentiment.polarity
    if sentiment_score > 0.1:
        return "🟢 Positive"
    elif sentiment_score < -0.1:
        return "🔴 Negative"
    else:
        return "🟡 Neutral"

# Fetch latest articles from Google News RSS (via NewsData.io for simplicity)
def fetch_top_news(topic, limit=5):
    url = f"https://newsdata.io/api/1/news?apikey={st.secrets['NEWSDATA_API_KEY']}&q={topic}&language=en"
    try:
        res = requests.get(url)
        articles = res.json().get("results", [])[:limit]
        return articles
    except Exception as e:
        st.error(f"Failed to fetch news: {str(e)}")
        return []

# UI
st.set_page_config(page_title="AI News Explorer 📰", layout="centered")
st.title("📰 AI News Explorer")
st.write("Explore the latest news on any topic, summarized and analyzed using Generative AI.")

topic = st.text_input("Enter a topic to explore:", value="AI")

if st.button("Explore News"):
    with st.spinner("Fetching and summarizing news..."):
        articles = fetch_top_news(topic)

        if not articles:
            st.warning("No news articles found.")
        else:
            for i, article in enumerate(articles, 1):
                title = article.get("title", "")
                content = article.get("description", "")
                full_text = f"{title}. {content}"
                
                summary = summarize_text_gemini(full_text, topic)
                sentiment = get_sentiment(summary)

                st.markdown(f"### {i}. {title}")
                st.markdown(f"**Summary:** {summary}")
                st.markdown(f"**Sentiment:** {sentiment}")
                st.markdown("---")
