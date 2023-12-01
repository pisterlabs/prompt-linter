from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import chromadb

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

chroma_dataset_name = "assistant_63"
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
vecdb = Chroma(
    client=chroma_client,
    collection_name=chroma_dataset_name,
    embedding_function=embeddings,
)

test_rep = "/Users/yuriy/Github/Vistrem/learning/chroma-peek"

docs = []
file_extensions = []

for dirpath, dirnames, filenames in os.walk(test_rep):
    for file in filenames:
        file_path = os.path.join(dirpath, file)

        if file_extensions and os.path.splitext(file)[1] not in file_extensions:
            continue

    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs.extend(loader.load_and_split())
    except Exception as e:
        print(f"Error while loading file {file_path}: {e}")
        continue

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_text = text_splitter.split_documents(docs)

# Embeddings
# vecdb.add_documents(splitted_text)

retriever = vecdb.as_retriever(
    distance_metric="cos",
    fetch_k=100,
    maximal_marginal_relevance=True,
    k=10,
)

model=ChatOpenAI()
qa = RetrievalQA.from_llm(model, retriever=retriever)

print(qa.run("What is the repository's name?"))

import streamlit as st
from streamlit_chat import message

st.title(f"Chat with Github Repository")

# Initialize the session state for placeholder messages.
if "generated" not in st.session_state:
	st.session_state["generated"] = ["i am ready to help you ser"]

if "past" not in st.session_state:
	st.session_state["past"] = ["hello"]

# A field input to receive user queries
input_text = st.text_input("", key="input")

# Search the databse and add the responses to state
user_input = ""
if user_input:
	output = qa.run(user_input)
	st.session_state.past.append(user_input)
	st.session_state.generated.append(output)

# Create the conversational UI using the previous states
if st.session_state["generated"]:
	for i in range(len(st.session_state["generated"])):
		message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
		message(st.session_state["generated"][i], key=str(i))