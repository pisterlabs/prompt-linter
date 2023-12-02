#PRE-REQUISITO - NA MESMA SEÇÃO DE USO TER REALIZADO nomic login <API-KEY>
from nomic import atlas
import os, sys
from langchain.schema import Document
import logging

if __name__ == '__main__':
    # Get the parent directory
    ROOT_PATH = os.getcwd()
    print(f'setting {ROOT_PATH} as ROOT_PATH')
    os.environ['APP_ROOT_PATH'] = ROOT_PATH

    print(f'appending {ROOT_PATH} to python path')
else:
    ROOT_PATH = os.getenv('APP_ROOT_PATH')

sys.path.append(ROOT_PATH)

import config.config as config
from document_load.document_loaders import load_document
from document_load.chunking import chunk_data
from embeddings_helper import *
import pandas as pd

os.environ['APP_ROOT_PATH'] = os.getcwd()
logging.basicConfig(level=config.LOG_LEVEL)

def generate_local_embeddings():

    file_path = os.path.join('data', 'CONTRATO.pdf')
    pages:list[Document] = load_document(file_path)
    chunks:list[Document] = chunk_data(pages) #Using defaults from config

    text_chunks = [chunk.page_content.replace(os.linesep, ' ') for chunk in chunks]
    dataframe_from_chunks = pd.DataFrame(columns=['page_content'], data=text_chunks)

    dataframe_from_chunks['embedding'] = dataframe_from_chunks['page_content'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    
    dataframe_from_chunks.to_csv(os.path.join('data', 'CONTRATO_EMBEDDINGS.csv'), index=False)
    print(dataframe_from_chunks)   

def push_to_atlas():
    dataframe_with_embeddings = pd.read_csv(os.path.join('data', 'CONTRATO_EMBEDDINGS.csv'))
    dataframe_with_embeddings = convert_embeddings_to_nparray(dataframe_with_embeddings)

    embeddings = list(dataframe_with_embeddings['embedding'])
    data = dataframe_with_embeddings[['page_content']].to_dict('records') 

    embeddings = np.array(embeddings)

    project = atlas.map_embeddings(
    embeddings=embeddings,
    data=data,
    name='Contrato'
    )



if __name__ == "__main__":
    push_to_atlas()