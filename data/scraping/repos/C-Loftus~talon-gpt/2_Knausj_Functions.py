import streamlit as st
import os , requests, pickle
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredURLLoader
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


# import env from one level up
import sys
sys.path.append("..")
import env


os.environ['OPENAI_API_KEY'] = env.API_KEY.openai

# App framework
st.title('🦜 Knausj GPT Creator')
prompt = st.text_input('I want a Knausj Talon function that I can say in order to: ', placeholder="open a new tab") 

# text_splitter = RecursiveCharacterTextSplitter(
# chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
# )
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# verbose name for loading screen shown to the user
@st.cache_data
def website_source_initialization2():

    urls = ["https://tararoys.github.io/small_cheatsheet"]
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    return data

with st.spinner('Starting up application... '):
    data = website_source_initialization2()
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()

    # verbose name for loading screen shown to the user
    @st.cache_data
    def database_initialization2():
        return FAISS.from_documents(docs, embeddings)
    vectorStore_openAI = database_initialization2()

st.divider()

# with open("knausj_faiss_store_openai.pkl", "wb") as f:
#     pickle.dump(vectorStore_openAI, f)
# # with open("knausj_faiss_store_openai.pkl", "rb") as f:
# #     VectorStore = pickle.load(f)
llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever())

# Show stuff to the screen if there's a prompt
with st.spinner('Querying the language model and searching docs. This will take a bit...'):
        outputs = chain({"question": prompt}, return_only_outputs=True)

        st.write("Answer: ", outputs['answer'])

        try:
            sources = outputs['sources']
        except:
            sources = outputs['source']

        if len(sources) > 0 and sources != None:
            st.write("Sources: ", outputs['sources'])

        st.success('Done!')