import streamlit as st
import pickle

# Set page title
st.set_page_config(page_title="Fake News Detector")

# 1. Load the saved model and vectorizer
@st.cache_resource # This keeps the model loaded so the app stays fast
def load_assets():
    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_assets()

# 2. UI Layout
st.title("🕵️‍♂️ Fake News Detector")
st.write("Enter a news headline below to see if the model classifies it as Real or Fake.")

user_input = st.text_input("News Headline:")

if st.button("Predict"):
    if user_input:
        # 3. Process and Predict
        data = [user_input]
        vectorized_input = vectorizer.transform(data)
        prediction = model.predict(vectorized_input)
        
        # 4. Display Result
        if prediction[0] == 1:
            st.error("🚨 Result: This looks like FAKE NEWS")
        else:
            st.success("✅ Result: This looks like REAL NEWS")
    else:
        st.warning("Please enter a headline first.")