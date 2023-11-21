# Import modules
import databutton as db
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


st.set_page_config("Multilingual Chat Bot 🤖", layout="centered")
st.sidebar.text("Resources")
st.sidebar.markdown(
    """ 
- [Multilingual Embedding Models](https://docs.cohere.com/docs/multilingual-language-models)
- [Multilingual Search with Cohere and Langchain](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Multilingual_Search_with_Cohere_and_Langchain.ipynb)
- [LangChain](https://python.langchain.com/en/latest/index.html)
- [ChatUI template from databutton](https://www.databutton.io)

You can find more similar apps of mine [here](https://databutton.com/v/lgzxq112)!

"""
)

# Cohere API Initiation
cohere_api_key = db.secrets.get(name="COHERE_API_KEY")

st.title("Multilingual Chat Bot 🤖")
st.info(
    "For your personal data! Powered by [cohere](https://cohere.com) + [LangChain](https://python.langchain.com/en/latest/index.html) + [Databutton](https://www.databutton.io) "
)

opt = st.radio("--", options=["Try the demo!", "Upload-own-file"])

# Unnecessary complication! -> Refractor required
pages = None
if opt == "Upload-own-file":
    # Upload the files
    uploaded_file = st.file_uploader(
        "**Upload a pdf or txt file :**",
        type=["pdf", "txt"],
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            doc = parse_txt(uploaded_file)
        else:
            doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
else:
    st.markdown(
        "Demo PDF : Steve Jobs' Stanford University commencement address as the example. "
        "[Link to download](https://docs.google.com/uc?export=download&id=1f1INWOfJrHTFmbyF_0be5b4u_moz3a4F)"
    )
    st.text("Quick Prompts to try (English | German):")
    
    st.code("What is the key lesson from this article?")
    st.code("Was ist die wichtigste Lehre aus diesem Artikel?")
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
        bot_message(message["content"], bot_name="Multilingual Personal Chat Bot")

if pages:
    # if uploaded_file.name.endswith(".txt"):

    # else:
    #     doc = parse_pdf(uploaded_file)
    # pages = text_to_docs(doc)

    with page_holder.expander("File Content", expanded=False):
        pages
    embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=cohere_api_key
    )
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
            llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
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
