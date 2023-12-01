# Refer to the Notebook below:
# https://github.com/cohere-ai/notebooks/blob/main/notebooks/Multilingual_Search_with_Cohere_and_Langchain.ipynb

# Import modules
import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import os
import random
import textwrap as tr
from text_load_utils import parse_txt, text_to_docs, parse_pdf, load_default_pdf
from df_chat import user_message, bot_message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

st.set_page_config("LLM based Chat Bot", layout="centered")
st.sidebar.text("Resources")
st.sidebar.markdown(
    """ 
- [Multilingual Embedding Models](https://docs.cohere.com/docs/multilingual-language-models)
- [Multilingual Search with Cohere and Langchain](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Multilingual_Search_with_Cohere_and_Langchain.ipynb)
- [LangChain](https://python.langchain.com/en/latest/index.html)


"""
)

with st.sidebar.expander(" 🛠️ Settings ", expanded=True):
    # Option to preview memory store
    MODEL = st.selectbox(
        label="Model",
        options=[
            "OPENAI",
            "Cohere",
        ],
    )
    if MODEL == "OPENAI":
        VERSION = st.selectbox(
        label="Version",
        options=[
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
        ],
    )

cohere_api_key = "enter your key here"
openai_api_key = "enter your key here" 

if MODEL == "OPENAI":
    embd_i = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm_i = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name=VERSION, verbose=False)

elif MODEL == "Cohere":
    embd_i = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=cohere_api_key
    )
    llm_i = Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key)

# Cohere API Initiation
#cohere_api_key = db.secrets.get(name="COHERE_API_KEY")


st.title("LLM Powered Chat Bot")
st.info(
    "For your personal data! Powered by [cohere](https://cohere.com) + [LangChain](https://python.langchain.com/en/latest/index.html) "
)

pages = load_default_pdf()

page_holder = st.empty()
# Create our own prompt template
prompt_template = """Text: {context}

Question: {question}

Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# Bot UI dump
# Session State Initiation
prompt = st.session_state.get("prompt", None)

if prompt is None:
    prompt = [{"role": "system", "content": prompt_template}]

# If we have a message history, let's display it
for message in prompt:
    if message["role"] == "user":
        user_message(message["content"])
    elif message["role"] == "assistant":
        bot_message(message["content"], bot_name="Chat Bot")

if pages:
    # if uploaded_file.name.endswith(".txt"):

    # else:
    #     doc = parse_pdf(uploaded_file)
    # pages = text_to_docs(doc)

    with page_holder.expander("File Content", expanded=False):
        pages
    embeddings = embd_i
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="Dot",
    )
    messages_container = st.container()
    question = st.text_input(
        "", placeholder="Type your message here", label_visibility="collapsed"
    )

    if st.button("Run", type="secondary"):
        prompt.append({"role": "user", "content": question})
        chain_type_kwargs = {"prompt": PROMPT}
        with messages_container:
            user_message(question)
            botmsg = bot_message("...", bot_name="Multilingual Personal Chat Bot")

        qa = RetrievalQA.from_chain_type(
            llm=llm_i,
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        answer = qa({"query": question})
        result = answer["result"].replace("\n", "").replace("Answer:", "")
        # with st.expander("Latest Content Source", expanded=False):
        #     sources = answer["source_documents"]
        # Update
        with st.spinner("Loading response .."):
            botmsg.update(result)
        # Add
        prompt.append({"role": "assistant", "content": result})

    st.session_state["prompt"] = prompt
else:
    st.session_state["prompt"] = None
    st.warning("No file found. Upload a file to chat!")
