# Import libraries
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
import re
from openai import OpenAI
import time
import openai

# Initialize the OpenAI client
client = OpenAI()

# Setup the directory for database
persist_directory = 'db'
embedding = OpenAIEmbeddings()

# Specify the unique persist directory for the vectordb
unique_persist_directory = 'db/9a81ee9dabdfbc3e9c94ad7cbaf6b79d'

# Initialize the vector database
vectordb = Chroma(persist_directory=unique_persist_directory, embedding_function=embedding)

# Create a retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# Initialize the LLM
llmm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llmm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True,
                                       verbose=True)

conversation_history = ""


def reformulate_query(original_query):
    try:
        response = openai.Completion.create(
            model="gpt-4-1106-preview",  # or the specific model you're using
            prompt="Reformulate this query into a more detailed and specific version: '" + original_query + "'",
            max_tokens=50  # Adjust the number of tokens as needed
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print("Error in generating response:", e)
        return original_query  # Fallback to original query in case of error


# Function to process the LLM response
def process_llm_response(llm_response):
    response = llm_response['result']
    sources = '\n\nSources:\n' + '\n'.join([source.metadata['source'] for source in llm_response["source_documents"]])
    return response + sources

# Import libraries and existing initialization code


# Modified function to handle chat and questions with conversation history and query reformulation
def handle_chat():
    global conversation_history
    while True:
        original_question = input("Ask a question: ")
        if original_question.lower() == 'exit':
            break

        # Reformulate the user's query
        detailed_question = reformulate_query(original_question)

        # Append the detailed question to the conversation history
        conversation_history += "User (detailed): " + detailed_question + "\n"

        # Retrieve documents considering the conversation history and detailed question
        docs = retriever.get_relevant_documents(conversation_history + detailed_question)
        if not docs:
            print("No relevant documents found.")
            continue

        # Generate response considering the conversation history and detailed question
        llm_response = qa_chain(conversation_history + detailed_question)
        response = process_llm_response(llm_response)

        # Append the response to the conversation history
        conversation_history += "System: " + response + "\n"

        print(response)

# Start the chat
handle_chat()

