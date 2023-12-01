import langchain
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

# Supplying a persist_directory will store the embeddings on disk
persist_directory = './api/db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectordb = Chroma(persist_directory=persist_directory, 
                embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
retriever.search_type

def process_user_input(query):
    results = {
        'answer': ''
    }
    try:
        #create the chain to answer questions 
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)
        response = qa_chain(query)
        results['answer'] = response['result']
    except Exception as e:
        print(f"An error occurred {e}")
        results["answer"]="Apologies, but I'm currently experiencing technical difficulties and I'm unable to assist you at the moment. Please try again later.\n\nIf the issue persists, please contact our support team for further assistance. Thank you for your understanding."
    return results