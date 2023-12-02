import streamlit as st

from typing import cast, List
from github import Github, PullRequest, File

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.header("Beacon > Codebase Analysis")
st.subheader("Kimley-Horn/SigOpsMetrics")

openai_token = st.secrets["openai_token"]
github_token = st.secrets["github_token"]

initial_question = "How many pull requests were submitted by each engineer?"
if "question" not in st.session_state:
    st.session_state.question = initial_question

if "previous_questions" not in st.session_state:
    st.session_state.previous_questions = [initial_question]

if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = []

@st.cache_data
def get_pull_requests():
    g = Github(login_or_token=github_token)
    repo = g.get_repo("KimleyHorn/SigOpsMetrics")
    pulls = repo.get_pulls(state="closed")
    return [{
        "number": pull.number,
        "title": pull.title,
        "body": pull.body,
        "author": pull.user.login,
        "created_at": pull.created_at,
        "merged_at": pull.merged_at,
    } for pull in
        pulls.get_page(0)[0:100]
    ]

pr_data = ""
for pull in get_pull_requests():
    if not pull["merged_at"]:
        continue
    pr_data += f"""
    Title: {pull['title']}
    Body: {pull['body']}
    Author: {pull['author']}
    Created At: {pull['created_at'].strftime("%Y-%m-%d")}
    Merge Date: {pull['merged_at'].strftime("%Y-%m-%d")}
    \n
    """

llm = OpenAI(temperature=0.2, openai_api_key=openai_token)

@st.cache_data
def get_llm_response(prompt):
    return llm(prompt)

def gen_related_questions(data, question):
    prompt = f"""

    Act as an AI assistant to a non-technical project manager.

    You are given a list of pull requests with the following information:
    Title, Body, Author, Merge Date 

    {data}

    You are asked the following question:

    {question}

    Your task is to generate a list of 3-5 related questions that you could answer.

    Return one question per line. Do not include bullet points or numbers, just the question.
    """
    return get_llm_response(prompt)

def get_answer(data, question):
    prompt = f"""

    Act as an AI assistant to a non-technical project manager.

    You are given a list of pull requests with the following information:
    Title, Body, Author, Merge Date 

    {data}

    Your task is to answer the following question about the pull requests above:

    {question}

    If the answer can be presented in a table, please format the answer in a markdown table.
    If the answer can be presented in a list, please format the answer in a markdown list.
    Otherwise, return the answer as a string with no quotes.
    If you don't know the answer, please return "I don't know".
    """
    return get_llm_response(prompt)

@st.cache_data
def get_recommended_questions(pull_request_data, question):
    related_questions = gen_related_questions(pull_request_data, question)
    return related_questions.strip().splitlines()

def on_input_change():
    if st.session_state.question not in st.session_state.previous_questions:
        st.session_state.previous_questions.append(st.session_state.question)

def on_button_click(response):
    st.session_state.question = response
    on_input_change()

def on_remove_click(response):
    st.session_state.previous_questions.remove(response)

with st.sidebar:
    st.title("Previous Questions")
    for response in reversed(st.session_state.previous_questions):
        st.button(label=response, key=f"prev-{response}", on_click=on_button_click, args=(response,))
        st.button(label="Remove", key=f"remove-{response}", on_click=on_remove_click, args=(response,))

    st.title("Recommended Questions")
    for question in get_recommended_questions(pr_data, st.session_state.question):
        st.button(label=question, key=f"rec-{question}", on_click=on_button_click, args=(question,))

st.text_input("Enter a question", key="question", on_change=on_input_change)


def q_and_a(question):
    st.markdown(get_answer(
        pr_data,
        question
    ))

q_and_a(st.session_state.question)