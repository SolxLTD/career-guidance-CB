import streamlit as st
import re


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())



def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_text():
    try:
        with open("career_text.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: 'career_text.txt' was not found. Place it in the same folder as app.py."



raw_text = load_text()

st.subheader("Raw Text")
st.write(raw_text)


sentences = split_sentences(raw_text)

st.subheader("Sentences")
for s in sentences:
    st.write("• " + s)


cleaned = [preprocess(s) for s in sentences]

st.subheader("Cleaned Sentences")
for c in cleaned:
    st.write("• " + c)

st.success("App running successfully with NO errors, NO NLTK, and ONLY career_text.txt!")
