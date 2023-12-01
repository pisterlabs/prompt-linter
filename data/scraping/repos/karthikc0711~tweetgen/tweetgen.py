import streamlit as st
import openai

openai.api_key = "YOUR_API_KEY"

def generate_text(prompt):
    response = openai.Completion.create(prompt=prompt, engine="text-davinci-003", temperature=0.7, max_tokens=139)
    return response["choices"][0]["text"]

st.title("TweetGen - AI Tweet Generator")

tweet_text = st.text_input("What you want to be said in the tweet:")
language = st.text_input("Language:")
keywords = st.text_input("Keywords:")
number_of_words = st.text_input("Number of words:")
hashtags = st.text_input("Hashtags:")
is_thread = st.checkbox("Is thread")
generate_tweet_button = st.button("Generate Tweet")

if generate_tweet_button:
    tweet_text = generate_text(f"Generate a professional tweet in {language} with {number_of_words} words about {keywords} with hashtags {hashtags}: {tweet_text}")
    tweets = tweet_text.split("\n")
    for tweet in tweets:
        st.write(tweet)
