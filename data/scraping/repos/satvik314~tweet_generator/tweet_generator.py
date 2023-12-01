import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
# from dotenv import load_dotenv
# load_dotenv()
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

tweet_template = """
Give me {number} tweets on {topic}.
"""

tweet_prompt = PromptTemplate(template = tweet_template, input_variables = ['number', 'topic'])

gpt3_model = OpenAI(model = "text-davinci-003", temperature = 1)

tweet_generator = LLMChain(prompt = tweet_prompt, llm = gpt3_model)

st.title("Tweet Generator 🐦")
st.subheader("🚀 Generate tweets on any topic")

topic = st.text_input("Topic")

number = st.number_input("Number of tweets", min_value = 1, max_value = 10, value = 1, step = 1)

if st.button("Generate"):
    tweets = tweet_generator.run(number = number, topic = topic)
    st.write(tweets)
    # for tweet in tweets:
    #     st.write(tweet)
