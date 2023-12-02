import pymongo
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId
import io
import fitz  # PyMuPDF
import openai
import requests

patient_number_to_search = 1

# Connect to MongoDBs
client = MongoClient("mongodb://localhost:27017/")
db = client[
    "patient_database"
]  # Replace 'patient_database' with your actual database name

# Access the 'patients' collection and the GridFS object
patients_collection = db["patients"]
fs = GridFS(db, collection="patient_files")


def get_pdf_content(patient_number):
    # Find the patient document using the patient number
    patient_data = patients_collection.find_one({"patient_number": patient_number})

    if patient_data:
        pdf_id = patient_data.get("pdf_id")

        if pdf_id:
            # Retrieve the PDF file from GridFS using the associated ObjectId
            pdf_file = fs.get(ObjectId(pdf_id))

            # Get the PDF content as bytes
            pdf_content = pdf_file.read()
            return pdf_content
        else:
            print("No PDF file associated with this patient.")
            return None
    else:
        print(f"No patient found with number {patient_number}")
        return None


# Example usage: Get PDF content for patient number 1
pdf_content = get_pdf_content(patient_number_to_search)

if pdf_content:
    # Open the PDF content directly using fitz
    doc = fitz.open(stream=io.BytesIO(pdf_content))
    all_text = ""
    for page in doc:
        all_text += page.get_text() + chr(12)
    # Perform actions with the doc object as needed
    # (e.g., extract text, perform analysis, etc.)

    # Close the document when done
    doc.close()
else:
    print("PDF content not found.")


##SUMMARIZER
client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are ChatGPT, a large language model that is trained to accept a patient's health record and return a summary specific details about this person in the format: current mediations names only||age,gender,known medical conditions,allergies. list it all out in 1 sentence comma seperated\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-02",
        },
        {"role": "user", "content": all_text},
    ],
    model="gpt-3.5-turbo",
    max_tokens=100,
)
print(chat_completion.choices[0].message.content)
result = chat_completion.choices[0].message.content

input_text = result
# Extract medicine names until the '||' delimiter
medicine_names = input_text.split("||")[0]

# Remove leading and trailing whitespaces
medicine_names = medicine_names.strip()

# Remove all spaces
medicine_names = medicine_names.replace(" ", "")


##GENERATING DRUG DRUG INTERACTIONS
resulting_list = medicine_names.split(",")
rxcui_list = ""

for drug_name in resulting_list:
    api_url = "https://rxnav.nlm.nih.gov/REST/drugs.json?name=" + drug_name
    response = requests.get(api_url)

    try:
        rxcui = response.json()["drugGroup"]["conceptGroup"][1]["conceptProperties"][1][
            "rxcui"
        ]
    except KeyError:
        # If the primary path is not available, try an alternative path
        try:
            rxcui = response.json()["drugGroup"]["conceptGroup"][2][
                "conceptProperties"
            ][1]["rxcui"]
        except KeyError:
            rxcui = "RxCUI not found"

    print(rxcui)
    rxcui_list = rxcui_list + rxcui + "+"
rxcui_list = rxcui_list[:-1]
print(rxcui_list)

api_url = "https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis=" + rxcui_list
response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()

    # Extracting and printing ONCHigh
    onchigh_set = next(
        (
            item
            for item in data.get("fullInteractionTypeGroup", [])
            if item.get("sourceName") == "ONCHigh"
        ),
        None,
    )
    print("Severity from ONCHigh:")

    if onchigh_set is not None:
        for interaction in onchigh_set.get("fullInteractionType", []):
            for interaction_pair in interaction.get("interactionPair", []):
                severity = interaction_pair.get("severity")
                if severity:
                    print(severity)
                    result = result + severity
                else:
                    print("No severity information available")
    else:
        print("No data found for ONCHigh")
    # Extracting and printing descriptions from DrugBank
    drugbank_interactions = [
        interaction
        for interaction in data.get("fullInteractionTypeGroup", [])
        if interaction.get("sourceName") == "DrugBank"
    ]

    print("\nDescriptions from DrugBank:")
    unique_descriptions = set()  # Using a set to store unique descriptions

    if not any(drugbank_interactions):
        print("No descriptions found for DrugBank interactions.")
    else:
        for interaction in drugbank_interactions:
            if "fullInteractionType" in interaction:
                for full_interaction in interaction["fullInteractionType"]:
                    if "interactionPair" in full_interaction:
                        for interaction_pair in full_interaction["interactionPair"]:
                            if "description" in interaction_pair:
                                unique_descriptions.add(interaction_pair["description"])

        # Printing unique descriptions
        for description in unique_descriptions:
            print(description)
            result = result + description

else:
    print("Request was not successful. Status code:", response.status_code)


print(result)
##THE AI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os
import openai

OPENAI_API_KEY = "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"
# Configure OpenAI
openai_api_base = ("https://api.openai.com/v1/",)
openai_api_key = (OPENAI_API_KEY,)
temperature = (0,)
engine = "gpt-3.5-turbo"

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # Change to 'cuda' for GPU if desired
    encode_kwargs=encode_kwargs,
)

# 1. Vectorize the sales response CSV data
loader = CSVLoader(file_path="D:\\Pranav\\Repository\\Python\\Records\\try1.csv")
documents = loader.load()

embeddings = model_norm
db = FAISS.from_documents(documents, embeddings)


# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=6)

    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/zephyr-7B-alpha-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, device_map="auto", trust_remote_code=False, revision="main"
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


input_text = "35 year old female is described 3mg ofloxaxin"
output = retrieve_info(input_text)
input = "35 year old female is described 3mg ofloxaxin"

prompt_template = f"""
You are an intelligent chatbot that predicts adverse drug reactions.
I will provide you a prescribed drugs, patient's age, sex, weight, the previous medical conditions, possible drug drug interactions which may or may not have dosage all as a single prompt also a list of known adverse reactions.
You will accurately predict what the list of possible adverse drug reactions.
1/ Response should be very similar or even identical to the past drug reactions, in terms of length, tone of voice, logical arguments, and other details

2/ If the prescription is not relevant enough, then try to mimic the style of possible adverse drug reaction

Below is a list of prompts with details of the patient and the drugs that are prescribed,adverse drug reactions,:
{input}
Here is a list of adverse drug reactions that occurred in similar scenarios:
{output}
Give the output in the following format in under 50 words just give the values without any tags:
Drug Name only,
list of adverse drug reactions not medical conditions with a short description with explanation,
risk level assessment as H for high and M for Medium and L for Low for the prescription and make that rating the last character in the output after a comma
"""

print("\n\n*** Generate:")


# 4. Retrieval augmented generation
# def generate_response(input):
#     output = retrieve_info(input)
#     response = chain.run(input=input, output=output)
#     return response

# aimodel()

print("Input Text:", input_text)

input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
reply = model.generate(
    inputs=input_ids,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    top_k=40,
    max_new_tokens=512,
)
print(tokenizer.decode(reply[0]))
