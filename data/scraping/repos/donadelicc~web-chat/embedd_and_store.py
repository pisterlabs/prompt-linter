import os
from dotenv import load_dotenv
from google.cloud import storage
import pickle
from io import BytesIO

from web_scrape import fetch_website_content, get_internal_links, scrape_all_pages


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def upload_to_gcs(data, bucket_name, blob_name):
    key_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if key_file is None:
        raise Exception("Google Cloud-nøkkelfilen er ikke definert i miljøvariablene")
    
    # Autentiser ved hjelp av nøkkelfilen
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =  key_file

    # Opprett en klient
    storage_client = storage.Client()

    # Få tak i bøtta
    bucket = storage_client.bucket(bucket_name)

    # Opprett en blob og last opp dataen
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type='application/octet-stream')


def embedd_and_store_text(text):

    load_dotenv()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )

    ## teksten burde kanskje formateres og forbedres før den blir sendt til en VectorStore
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    pkl_bytes = VectorStore.serialize_to_bytes()  # Serialiserer FAISS-indeksen

    bucket_name = 'chatty1'  # Erstatt med ditt bøttenavn
    blob_name = 'COAX_web_content.pkl'

    # Laste opp til Google Cloud Storage
    upload_to_gcs(pkl_bytes, bucket_name, blob_name)


text = scrape_all_pages("https://www.coax.no/") ## få GPT til å lage en ny og mer sammenhengende tekst?
#print(text)
embedd_and_store_text(text)