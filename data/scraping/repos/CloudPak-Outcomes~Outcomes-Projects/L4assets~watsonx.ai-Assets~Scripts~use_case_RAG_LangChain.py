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

from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# WML python SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

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

def get_lang_chain_model(model_type,max_tokens,min_tokens,decoding,temperature):

    base_model = get_model(model_type,max_tokens,min_tokens,decoding,temperature)
    langchain_model = WatsonxLLM(model=base_model)

    return langchain_model


def main():

    # Get the API key and project id and update global variables
    get_credentials()

    # Test answering questions based on the provided .pdf file
    # question = "What is Generative AI?"
    # question = "What does it take to build a generative AI model?"
    question = "What are the limitations of generative AI models?"
    # Provide the path relative to the dir in which the script is running
    # In this example the .pdf file is in the same directory
    file_path = "./Generative_AI_Overview.pdf"

    answer_questions_from_doc(api_key, watsonx_project_id, file_path, question)

def answer_questions_from_doc(request_api_key, request_project_id, file_path, question):

    # Update the global variable
    globals()["api_key"] = request_api_key
    globals()["watsonx_project_id"] = request_project_id

    # Specify model parameters
    model_type = "meta-llama/llama-2-70b-chat"
    max_tokens = 300
    min_tokens = 100
    decoding = DecodingMethods.GREEDY
    temperature = 0.7

    # Get the watsonx model that can be used with LangChain
    model = get_lang_chain_model(model_type, max_tokens, min_tokens, decoding, temperature)

    loaders = [PyPDFLoader(file_path)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)).from_loaders(loaders)

    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=index.vectorstore.as_retriever(),
                                        input_key="question")

    # Invoke the chain
    response_text = chain.run(question)

    # print model response
    print("--------------------------------- Generated response -----------------------------------")
    print(response_text)
    print("*********************************************************************************************")

    return response_text

# Invoke the main function
if __name__ == "__main__":
    main()
