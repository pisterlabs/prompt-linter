import streamlit as st
from utils.peek import ChromaPeek
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
load_dotenv()

PATH = "data/db"
peeker = ChromaPeek(PATH)
open_ai = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

st.set_page_config(page_title="RAG System",
                   layout="wide",
                   page_icon='👀'
                   )

st.title("RAG System")


def get_prompt(question, context):
    return [
        {"role": "system", "content": "Sie sind ein hilfreicher Assistent."},
        {
            "role": "user",
            "content": f"""Beantworten Sie die folgende Frage basierend auf dem Kontext.
    Frage: {question}\n\n
    Kontext: {context}\n\n
    Antwort:\n""",
        },
    ]


@st.cache_data()
def get_collections():
    return peeker.get_collections()


@st.cache_data()
def generate_answer(query, context, generator, temperature, top_p, max_tokens):
    messages = get_prompt(query, context)
    response = open_ai.chat.completions.create(
        model=generator,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response, messages[1]['content']


embeddings_collection = st.sidebar.selectbox('embeddings collection',
                                             options=get_collections()
                                             )
top_k = st.sidebar.number_input('top_k', min_value=1, max_value=10, value=3)

generator = st.sidebar.selectbox('generator model',
                                 options=['gpt-3.5-turbo-1106',
                                          'gpt-4-1106-preview',
                                        #   'LeoLM/leo-mistral-hessianai-7b-chat'
                                          ]
                                 )
temperature = st.sidebar.slider('temperature', min_value=0.0, max_value=2.0, value=0.85, step=0.01)
top_p = st.sidebar.slider('top_p', min_value=0.0, max_value=1.0, value=0.9, step=0.01)
max_tokens = st.sidebar.number_input('max_tokens', min_value=1, max_value=1000, value=512, step=50)
use_context = st.sidebar.checkbox('show context', value=False)
query = st.text_input("Enter Query", placeholder="query")
if st.button('Answer') and query:
    relevant_docs = peeker.query(query, embeddings_collection, top_k, dataframe=False)
    context = ''
    for metadata, document in zip(relevant_docs['metadatas'], relevant_docs['documents']):
        context += f"{metadata['document_name']}.\n{document}\n\n"
    response, question = generate_answer(query, context, generator, temperature, top_p, max_tokens)
    st.write(response.choices[0].message.content)
    st.dataframe(pd.DataFrame(relevant_docs), use_container_width=True)
    if use_context:
        st.write(question)
