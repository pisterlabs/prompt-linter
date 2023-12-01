#----------------------------------------- Import packages -----------------------------------------
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from credentials import OPENAI_API_KEY, ASSEMBLYAI_API_KEY

import requests
import time

from credentials import OPENAI_API_KEY
import pandas as pd
import datetime
from dotenv import load_dotenv
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
#---------------------------------------------------------------------------------------------------

#-------------------------------------- Import Keys ------------------------------------------------
base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": ASSEMBLYAI_API_KEY 
}

load_dotenv()
#---------------------------------------------------------------------------------------------------


#--------------------------------------- Voicenote to text -----------------------------------------

def voicenote(upload_url):    
  data = {
    "audio_url": upload_url # You can also use a URL to an audio or video file on the web
  }
  url = base_url + "/transcript"
  response = requests.post(url, json=data, headers=headers)

  transcript_id = response.json()['id']
  polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

  while True:
    transcription_result = requests.get(polling_endpoint, headers=headers).json()

    if transcription_result['status'] == 'completed':
      return transcription_result['text']

    elif transcription_result['status'] == 'error':
      raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

    else:
      time.sleep(3)
#---------------------------------------------------------------------------------------------------

#---------------------------------------- Data Collection ------------------------------------------

def data_collection(answer):
  num_media = request.values.get('NumMedia', '')
  sms_id = request.values.get('SmsSid', '')
  wa_id = request.values.get('WaId', '')
  body = request.values.get('Body', '')
  timestamp = datetime.datetime.now()
  answer = answer.strip()


  # store into a dataframe
  # Sample data to append
  data_to_append = {
      'NumMedia': [num_media],
      'SmsSid': [sms_id],
      'WaId': [wa_id],
      'Body': [body],
      'Answer': [answer],
      'Timestamp': [timestamp]
  }

  # Create a new DataFrame from the data
  new_data_df = pd.DataFrame(data_to_append)

  # Read the existing DataFrame from the CSV file
  existing_df = pd.read_csv("Data_Lake/messages.csv")

  # Concatenate the existing DataFrame and the new DataFrame
  df = pd.concat([existing_df, new_data_df], ignore_index=True)

  # Save the combined DataFrame to a CSV file
  df.to_csv("Data_Lake/messages.csv", index=False)
  
 #---------------------------------------------------------------------------------------------------

# --------------------------------------- CONNECT TO VECTORDATABASE  ------------------------------------------------ #
def create_vectordb(url):
    # Load URL
    loader = WebBaseLoader(url)
    docs_url = loader.load()

    # Initialize a list to store the PDFs
    docs_pdf = []

    # File names and corresponding loader instances
    file_loader_pairs = [
        ("./docs/08009636_Sant Miquel_Resum de l'EDC.pdf", None),
        ("./docs/DM_Dir_NOF_CSM_14_set_2023.pdf", None),
        ("./docs/DM_DIR_PEC_JUNY_23.pdf", None),
        ("./docs/NOF_digital.pdf", None)
    ]

    # Load data for each file
    for i, (file_name, _) in enumerate(file_loader_pairs):
        loader = PyPDFLoader(file_name)
        # Access the page_content attribute for each Document in the list
        file_loader_pairs[i] = (file_name, [doc.page_content for doc in loader.load()])

    # Extract data for merging
    merged_docs = [content for _, data in file_loader_pairs for content in data if content]

    # Ensure merged_docs is a list of strings
    if not all(isinstance(doc, str) for doc in merged_docs):
        raise ValueError("Merged documents should be a list of strings.")

    # Split text
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
        separators=["\n\n", "\n", "(?<=\. )", " ", "  "]
    )
    splits = []
    for doc in merged_docs:
        splits.extend(r_splitter.split_text(doc))

    # Create Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "python-index"

    # Create Vector Database using Pinecone
    vectordb = Pinecone.from_texts(texts=splits, embedding=embeddings, index_name=index_name)

    return vectordb
