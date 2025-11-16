import streamlit as st
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Download NLTK resources ---
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


with open("app/career_text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().lower()

def preprocess(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

sentences = sent_tokenize(raw_text)
clean_sentences = [preprocess(s) for s in sentences]


def get_most_relevant_sentence(user_query):
    user_query = preprocess(user_query)
    all_sentences = clean_sentences + [user_query]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_sentences)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = np.argmax(similarity)
    return sentences[index], similarity[0][index]


def chatbot(query):
    response, score = get_most_relevant_sentence(query)
    if score < 0.1:
        return "I'm not sure, but you can explore career resources online for more info."
    else:
        return response


def main():
    st.title("🎓 Career Guidance Chatbot")
    st.write("Ask me questions about careers, skills, and personal development!")

    user_input = st.text_input("You:", "")
    if st.button("Ask"):
        if user_input:
            reply = chatbot(user_input)
            st.text_area("Chatbot:", reply, height=150)

if __name__ == "__main__":
    main()
