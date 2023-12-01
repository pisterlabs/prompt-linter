# import streamlit module
import streamlit as st

# import langchain modules

from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')

# Title
st.title('🦜🔗 Ask the Doc App')

# Document loader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# Index that wraps above steps
index = VectorstoreIndexCreator().from_loaders([loader])

# Question-answering
question = "What is Task Decomposition?"
response = index.query(question)

st.write(response)