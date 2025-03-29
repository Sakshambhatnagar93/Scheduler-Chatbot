import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("conversation_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("AI Chatbot")

# Function to get response
def get_response(user_input):
    X_input = vectorizer.transform([user_input])
    return model.predict(X_input)[0]

# User input
user_input = st.text_input("Enter your message:")
if st.button("Get Response"):
    if user_input:
        response = get_response(user_input)
        st.write("Bot Response:", response)
    else:
        st.write("Please enter a message.")
