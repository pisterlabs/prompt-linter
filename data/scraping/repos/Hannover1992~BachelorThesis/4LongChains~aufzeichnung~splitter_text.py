import os

from langchain import text_splitter
from langchain.llms import OpenAI
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# llm = OpenAI(temperature=0.9)

from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma


def load_data(filename, chunk_size=5000):
    with open(filename, 'r') as file:
        text = file.read()
    return text
    # # Create a list of Document objects, each containing a chunk of text
    # return [Document(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]




embeddings = OpenAIEmbeddings()

from langchain.text_splitter import PythonCodeTextSplitter
with open('Wissenschaftliche_Methoden/all_txt.txt', 'r') as file:
    text = file.read()

# Split text into words
python_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=0)
docs = python_splitter.create_documents([text])
# docs = text_splitter.split_documents(documents)
#

store = Chroma.from_documents(docs, embeddings)

query = "Wie Schreibt man en Abstract?"
# docs = store.similarity_search(query)



# prompt = st.text_input('Niebo gwazdziste nademna, prawo moralne we mnie. A ty, czym jesteś?Jestem twoją wolnością. Jestem tym, co masz, czego się trzymasz, czym możesz zdecydować i dążyć do tego, co uważasz za słuszne. Jestem tym, co możesz zmienić i wpłynąć na życie innych. Jestem tym, co jest w twojej ręce.')
prompt = 'how ot wirte an abstract?'
search = store.similarity_search_with_score(prompt)
st.write(search[0][0].page_content)
# if prompt:
#     with st.expander('Document Similarity Search'):
#         search = store.similarity_search_with_score(prompt)
#         st.write(search[0][0].page_content)