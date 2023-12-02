from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from PIL import Image
import os
import streamlit as st
from dotenv import load_dotenv

# load in .env variables 
load_dotenv()

def config_keys():
    # set api keys for AOAI and Azure Search
    os.environ['OPENAI_API_VERSION'] = os.getenv('AZURE_OPENAI_VERSION')
    os.environ['OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_KEY')
    os.environ['OPENAI_API_BASE'] = os.getenv('AZURE_OPENAI_ENDPOINT') 
    os.environ['OPENAI_EMBEDDING_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')
    os.environ['AZURE_COGNITIVE_SEARCH_SERVICE_NAME'] = os.getenv('AZURE_COGNITIVE_SEARCH_SERVICE_NAME')
    os.environ['AZURE_COGNITIVE_SEARCH_API_KEY'] = os.getenv('AZURE_COGNITIVE_SEARCH_API_KEY')
    os.environ['AZURE_COGNITIVE_SEARCH_INDEX_NAME'] = os.getenv('AZURE_COGNITIVE_SEARCH_INDEX_NAME')


def main():
    # Streamlit config 
    st.title("Demo - Azure OpenAI & Cognitive Search Embeddings")
    image = Image.open('image_logo2.png')
    st.image(image, caption = '')
    st.write('This program is designed to chat over your files in Azure Cognitive Search. \
                Be specific and clear with the questions you ask. \
                    Welcome to CHATGPT over your own data !!')
    if 'generated' not in st.session_state:
        st.session_state.generated = []    
    if 'past' not in st.session_state:
        st.session_state.past = []  

    # create your LLM and embeddings. Will be conifuring 'azure' in the openai_api_type parameter.
    llm = AzureChatOpenAI(  
                                deployment_name = "gpt-35-turbo",  
                                openai_api_type = "azure",  
                                model = "gpt-35-turbo",  
                                temperature=0.7, 
                                max_tokens=200
                                ) 

    embeddings = OpenAIEmbeddings(chunk_size=1, openai_api_type="azure") 

    # ask for the user query 
    query = st.text_input("Enter a search query: ", key='search_term', placeholder="")

    if query:
        st.session_state.past.append(query)

        # set up Azure Cognitive Search to retrieve documents
        # top_k = 1: we only want first related doc
        retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=1)

        # get the relevant document from Azure Cognitive Search that are only relevant to the query being asked
        docs = retriever.get_relevant_documents(query)

        # create embedding from the document retrieved and place in a FAISS vector database
        db = FAISS.from_documents(documents=docs, embedding=embeddings)

        # set up the chain that will feed the retrieved document to the LLM
        chain = RetrievalQA.from_chain_type(llm=llm, retriever = db.as_retriever(), chain_type="stuff")

        # run the chain on the query asked
        response = chain.run(query)
        st.session_state.generated.append(response)

        with st.expander('Vector Search'):
            for i in range(len(st.session_state.generated)-1, -1, -1):
                st.info(st.session_state.past[i])
                st.success(st.session_state.generated[i])

if __name__ == '__main__':
    config_keys()
    main()