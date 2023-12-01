"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a Question and Answer use case for a provided document


# Install the wml api your Python env prior to running this example:
# pip install ibm-watson-machine-learning

# Install chroma
# pip install chromadb

# In some envrironments you may need to install chardet
# pip install chardet

IMPORTANT: Be aware of the disk space that will be taken up by documents when they're loaded into
chromadb on your laptop. The size in chroma will likely be the same as .txt file size
"""

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction

# WML python SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

FILE_TYPE_TXT = "txt"
FILE_TYPE_PDF = "pdf"

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# These global variables will be updated in get_credentials() function
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""

def get_credentials():

    load_dotenv()
    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

# The get_model function creates an LLM model object with the specified parameters

def get_model(model_type,max_tokens,min_tokens,decoding,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model


# Embedding function
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):

    MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()

def create_embedding(file_path,file_type,collection_name):

    if file_type == FILE_TYPE_TXT:
        loader = TextLoader(file_path,encoding="1252")
        documents = loader.load()
    elif file_type == FILE_TYPE_PDF:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(type(texts))

    # Load chunks into chromadb
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name,embedding_function=MiniLML6V2EmbeddingFunction())
    collection.upsert(
        documents=[doc.page_content for doc in texts],
        ids=[str(i) for i in range(len(texts))],  # unique for each doc
    )

    return collection

def create_prompt(file_path, file_type, question, collection_name):

    # Create embeddings for the text file
    collection = create_embedding(file_path,file_type,collection_name)

    # Query relevant information
    # You can try retrieving different number of chunks (n_results)
    relevant_chunks = collection.query(
        query_texts=[question],
        n_results=3,
    )

    context = "\n\n\n".join(relevant_chunks["documents"][0])
    # Please note that this is a generic format. You can change this format to be specific to llama
    prompt = (f"{context}\n\nPlease answer a question using this "
              + f"text. "
              + f"If the question is unanswerable, say \"unanswerable\"."
              + f"{question}")

    return prompt

def main():

    # Get the API key and project id and update global variables
    get_credentials()

    # Ask a question relevant to the info in the document
    # You can try asking different questions
    question = "What did the president say about corporate tax?"
    # question = "What did the president say about jobs?"
    # question = "What did the president say about inflation?"
    # Provide the path relative to the dir in which the script is running
    # In this example the .txt file is in the same directory
    file_path = "./state_of_the_union.txt"

    collection_name = "state_of_the_union"

    # Test answering questions based on the provided .txt file
    answer_questions_from_doc(api_key,watsonx_project_id,file_path,FILE_TYPE_TXT,question,collection_name)

    # Test answering questions based on the provided .pdf file
    question = "How can you build a Generative AI model?"
    # question = "What are the limitations of generative AI models?"
    # Provide the path relative to the dir in which the script is running
    # In this example the .pdf file is in the same directory
    file_path = "./Generative_AI_Overview.pdf"

    collection_name = "generative_ai_doc"

    answer_questions_from_doc(api_key, watsonx_project_id, file_path, FILE_TYPE_PDF,question,collection_name)

def answer_questions_from_doc(request_api_key, request_project_id, file_path,file_type,question,collection_name):

    # Update the global variable
    globals()["api_key"] = request_api_key
    globals()["watsonx_project_id"] = request_project_id

    # Specify model parameters
    model_type = "meta-llama/llama-2-70b-chat"
    max_tokens = 300
    min_tokens = 100
    decoding = DecodingMethods.GREEDY
    temperature = 0.7

    # Get the watsonx model
    model = get_model(model_type, max_tokens, min_tokens, decoding, temperature)

    # Get the prompt
    complete_prompt = create_prompt(file_path, file_type, question, collection_name)

    # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")
    print("*** Prompt:" + complete_prompt + "***")
    print("----------------------------------------------------------------------------------------------------")

    generated_response = model.generate(prompt=complete_prompt)
    response_text = generated_response['results'][0]['generated_text']

    # print model response
    print("--------------------------------- Generated response -----------------------------------")
    print(response_text)
    print("*********************************************************************************************")

    return response_text


# Invoke the main function
if __name__ == "__main__":
    main()
