
import pinecone 
import streamlit as st 
from langchain.llms import OpenAIChat
from langchain.chains import VectorDBQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Pinecone
embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key="xxx",  
    environment="xxx"  
)
index_name = "sf-building-codes"
docsearch_sf_building_pinecone = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings)

# App
st.sidebar.image("Img/construction_bot.jpeg")
st.header("`Govt Co-Pilot`")
st.info("`Hello! I am a ChatGPT connected to the San Francico building code.`")
query = st.text_input("`Please ask a question:` ","At what size do I need a permit for a storage shed in my backyard?")
llm = OpenAIChat(temperature=0)
chain_pinecone_building_cgpt = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=docsearch_sf_building_pinecone)
result = chain_pinecone_building_cgpt.run(query)
st.info("`%s`"%result)