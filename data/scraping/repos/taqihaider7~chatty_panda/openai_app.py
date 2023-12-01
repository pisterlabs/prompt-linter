import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from PIL import Image
from langchain.vectorstores import Pinecone
import pinecone
from tqdm import tqdm
from langchain.embeddings import CohereEmbeddings
import cohere


API = st.secrets["API"]
pinecone_API = st.secrets["PINECONE_API"]
cohere_API = st.secrets["COHERE_API"]

# Set up Streamlit app
image= Image.open("app_banner.png")
st.image(image, use_column_width=True)
st.markdown(" **:red[Note :]** :blue[This App is a Prototype and Model is trained on limited Data of Emirates Airline]     :green[...Thanks for attention. !] ")
# Load and process documents
# st.write("Loading and processing documents...")
# loader = DirectoryLoader("data", glob="**/*.txt",loader_cls=TextLoader, use_multithreading=True, show_progress=True)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.split_documents(documents)


# initailizing the Pincone vector Database

pinecone.init(
      api_key=pinecone_API, environment="asia-southeast1-gcp")

embeddings= CohereEmbeddings(model = "embed-multilingual-v2.0", cohere_api_key=cohere_API)

# Set up question-answering model
st.write("Setting up question-answering model...")
# docsearch = Chroma.from_documents(texts, embeddings)
docsearch = Pinecone.from_existing_index(index_name="chatty-panda-embeddings", embedding = embeddings)
qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=API), chain_type="map_reduce", retriever=docsearch.as_retriever())


# Query the Data 
# st.markdown("# This App has been paused due to OpenAI API issue, and will be resumed once issue is resolved")
st.write("Enter your question below and click 'Ask' to get an answer:")
query = st.text_input("Question: ")
if st.button("Ask"):
    if not query:
        st.warning("Please Enter the Question.")
    else:
        st.write("Searching for Answer...")
        answer = qa.run(query)
        if answer:
            st.success(f"Answer: {answer}")
        else:
            st.error("Sorry, No Answer Was Found.")

# Set page footer
footer = """
<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made with ❤️: by Menlo Park Boys</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)