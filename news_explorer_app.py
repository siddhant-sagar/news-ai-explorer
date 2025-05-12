import streamlit as st
import pandas as pd
import datetime
from textblob import TextBlob
import google.generativeai as genai
import os

# ---- Gemini API Setup ----
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.0-flash")

# ---- Fetch News from Gemini ----
def fetch_news_from_gemini(topic):
    prompt = f"""Search and summarize the top 5 latest news headlines and summaries related to "{topic}" from trusted sources. 
    Keep it concise and informative, use bullet points."""
    try:
        response = model.generate_content(prompt)
        raw_output = response.text
        # Split bullet points
        summaries = [line.strip("-• ") for line in raw_output.strip().split("\n") if line.strip()]
        return summaries
    except Exception as e:
        st.error(f"Failed to fetch news from Gemini: {e}")
        return []

# ---- Sentiment Classification ----
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# ---- Log Analytics to CSV ----
def log_query(topic, total_articles, timestamp):
    df = pd.DataFrame([{
        "topic": topic,
        "articles_found": total_articles,
        "timestamp": timestamp
    }])
    df.to_csv("analytics_log.csv", mode='a', header=not pd.io.common.file_exists("analytics_log.csv"), index=False)

# ---- Streamlit App ----
st.set_page_config(page_title="AI News Summarizer", layout="centered")
st.title("🧠 AI News Summarizer with Sentiment Filter")

topic = st.text_input("Enter a topic (e.g., AI, Climate Change, Bitcoin):")
apply_filter = st.checkbox("Filter for positive sentiment only")

if st.button("Summarize News"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Fetching news using Gemini..."):
            news_items = fetch_news_from_gemini(topic)
            results = []
            for item in news_items:
                sentiment = classify_sentiment(item)
                if apply_filter and sentiment != "positive":
                    continue
                results.append((item, sentiment))

            timestamp = datetime.datetime.now().isoformat()
            log_query(topic, len(results), timestamp)

        if results:
            st.success(f"Showing {len(results)} summaries:")
            for i, (summary, sentiment) in enumerate(results, 1):
                st.markdown(f"**{i}.** {summary}")
                st.caption(f"🧭 Sentiment: `{sentiment}`")
        else:
            st.info("No summaries match the sentiment filter.")
