from llama_index import (
    download_loader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    QuestionAnswerPrompt,
)
from langchain import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")


def main(query, twitter_handles):
    TwitterTweetReader = download_loader("TwitterTweetReader")

    loader = TwitterTweetReader(bearer_token=twitter_bearer_token)
    documents = loader.load_data(twitterhandles=twitter_handles.split(","))

    QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question in bullet point format with e new line after each point: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name="text-davinci-003", openai_api_key=openai_api_key
        )
    )
    index = GPTSimpleVectorIndex(documents)
    response = index.query(query, text_qa_template=QA_PROMPT)
    print(response)
    return response


st.header("Twitter Qs")

twitter_handles = st.text_input("Enter the twitter handles you want to search")
user_input = st.text_input("Ask a question about the twitter handles")

if st.button("Find out"):
    st.markdown(main(user_input, twitter_handles=twitter_handles))
