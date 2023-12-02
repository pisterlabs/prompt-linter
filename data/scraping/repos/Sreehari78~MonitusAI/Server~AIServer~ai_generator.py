# ai_generator.py
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os
# Your existing AI model and response generation logic

OPENAI_API_KEY = "OPENAI_API_KEY"
# Configure OpenAI
openai_api_base = ("https://api.openai.com/v1/",)
openai_api_key = OPENAI_API_KEY,
temperature = (0,)
engine = "gpt-3.5-turbo-1106"

# 1. Setup FAISS vectorizer

llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1/",
    openai_api_key= OPENAI_API_KEY,
    temperature=0,
    # engine="gpt-3.5-turbo"
)

template = """
You are an intelligent chatbot that predicts adverse drug reactions. Given a prescription and patient details, predict possible adverse reactions.

Prescription and Patient Information:
{input}

List of Adverse Drug Reactions in Similar Scenarios:
{output}

Output Format (in under 30 words):
Drug Name only||Short description of possible interactions and allergies for the prescribed drug only||List any 7 drug side effects||Risk level as H for high, M for Medium, and L for Low only (e.g., Aspirin||No interactions||Headache||M)
The entire output should be a single sentence with no line breaks or extra spaces.
"""

prompt = PromptTemplate(template=template, input_variables=["input", "output"])

# Create an LLMChain instance with the prompt and ChatOpenAI instance
chain = LLMChain(prompt=prompt, llm=llm)

def generate_responses(input_text, faiss_vectorizer):
    generatedresponses =[]
    def generate_response_for_medicine(input_text):
        # print(f"Input Text for {medicine}: {input_text}")

        # Assuming retrieve_info and chain are defined in the global scope
        output = retrieve_info(input_text)
        result = chain.run(input=input_text, output=output)  # Fixed input parameter

        print("\n\nGenerated Response: " + result + "\n")
        generatedresponses.append(result)
        print(result)

    #Function for similarity search
    def retrieve_info(query):
        similar_response = faiss_vectorizer.similarity_search(query, k=5)
        page_contents_array = [doc.page_content for doc in similar_response]
        return page_contents_array

    with ThreadPoolExecutor() as executor:
        executor.map(generate_response_for_medicine, input_text)
    # for medicine in input_text:
        # generate_response_for_medicine(medicine)
        
    return generatedresponses
