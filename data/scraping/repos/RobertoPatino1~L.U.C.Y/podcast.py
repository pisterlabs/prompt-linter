import requests
from bs4 import BeautifulSoup
import os
import pickle
import json
import subprocess

import sys
sys.path.append('./')
import podcast_downloader.helpers as helpers
from podcast_downloader.helpers import slugify

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import re

from functools import cache

DATA_PATH = './podcast_downloader/transcripts'

@cache
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                       model_kwargs={'device': 'cpu'})
    return embeddings


embeddings = load_embeddings()
base_dir = './podcast_downloader'

class Podcast:
    def __init__(self, name:str, rss_feed_url:str):
        # Definir atributos de clase
        self.name = name
        self.rss_feed_url = rss_feed_url
        
        # Definir directorios de clase)
        self.download_directory = f'{base_dir}/downloads/{slugify(name)}'
        self.transcription_directory = f'{base_dir}/transcripts/{slugify(name)}'

        self.db_faiss_path = f'vectorstore/db_faiss/{slugify(name)}'

    
        # Crear directorios de clase
        for dir in [self.download_directory, self.transcription_directory]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def search_items(self, message, **kwargs):
        matched_podcasts = []
        # Obtener los items del podcast
        items = self.get_items()
        # Obtener los embeddings del podcast respecto a sus descripciones
        store_name = slugify(self.name)
        path = helpers.get_desc_emb_dir()
        db_description_embeddings = get_embeddings(store_name, path, embeddings, host_documents=False)['faiss_index']
        # Instanciar retriever
        retriever = db_description_embeddings.as_retriever(search_kwargs=kwargs)
        # Obtener descripciones que se asimilen al mensaje
        docs = retriever.get_relevant_documents(message)
        # Obtener los episodios indexados por título
        doc_descriptions = [x.page_content for x in docs]
        items_descriptions = [self.get_cleaned_description(x) for x in items]

        for doc_description in doc_descriptions:
            ind_description = items_descriptions.index(doc_description)
            matched_podcasts += [items[ind_description]]

        return matched_podcasts
    
    def update_description_embeddings(self):
        # Obtener episodios del podcast
        items = self.get_items()
        
        # Obtener los embeddings del podcast respecto a sus descripciones
        store_name = slugify(self.name)
        path = helpers.get_desc_emb_dir()
        metadata = get_embeddings(store_name, path, embeddings, host_documents=False)
        db_descriptions = metadata['texts'] 

        to_add = []
        for item in items:
            description = self.get_cleaned_description(item)
            if description not in db_descriptions:
                to_add += [description]

        if len(to_add) > 0:
            # Agregar description embedding 
            update_embeddings(to_add,store_name, path, embeddings, host_documents=False)        

    # Paragraph embeddings methods    
    def update_paragraph_embeddings(self, title, url):
        slugified_episode = slugify(title)
        transcripts_paths = os.listdir(self.transcription_directory)
        if f'{slugified_episode}.txt' not in transcripts_paths:
            self.generate_transcript(slugified_episode, url)

            db = None

            loader = TextLoader(f'{self.transcription_directory}/{slugified_episode}.txt')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            for doc in docs:
                doc.metadata['podcast'] = self.name
                doc.metadata['episode'] = title
            
            if not os.path.exists(self.db_faiss_path):
                db = FAISS.from_documents(docs, embeddings)
            else:
                db =  FAISS.load_local(self.db_faiss_path, embeddings)
                db.add_documents(documents=docs)
                 
            db.save_local(self.db_faiss_path)

    def generate_transcript(self, episode_path, url):
        # Obtener el path del transcript
        download_episode_path = f'{self.download_directory}/{episode_path}.mp3'
        print("Download path: ", download_episode_path)
        # Post de la metadata del podcast a obtener el transcript
        episode_metadata_json = {'url': url, 'download_episode_path': download_episode_path}
        with open(f'{base_dir}/podcast_metadata.json', 'w') as f:
            json.dump(episode_metadata_json, f)
        
        # subprocess.run([f'{base_dir}/run_all.sh'])
        subprocess.call(['python', f'{base_dir}/download_podcasts.py'])
        subprocess.call(['python', f'{base_dir}/transcriptions.py'])
        
    # Helpers methods
    def get_items(self):
        page = requests.get(self.rss_feed_url)
        soup = BeautifulSoup(page.text, 'xml')
        return soup.find_all('item')
    
    def get_cleaned_description(self, item):
        raw_description = item.find('description').text
        bs_description = BeautifulSoup(raw_description, 'html.parser')
        description = "\n".join([p.get_text(strip=True) for p in bs_description.find_all('p')])
        return description
    
    def get_language(self):
        page = requests.get(self.rss_feed_url)
        soup = BeautifulSoup(page.text, 'xml')
        return soup.find('language').text
    
    def get_ts_language(self):
        language = self.get_language()
        return convert_language_variable(language)

# Embeddings methods
def update_embeddings(texts_to_add:list, store_name:str, path:str, embeddings:HuggingFaceEmbeddings, host_documents:bool):
    # Obtener el vectordb
    metadata = get_embeddings(store_name, path, embeddings, host_documents=host_documents)
    vectorStore = metadata['faiss_index']
    # Agregar los textos al vectordb
    vectorStore.add_texts(texts_to_add)
    texts = metadata['texts'] + texts_to_add
    # Create a dictionary containing the metadata
    metadata = {
        'store_name': store_name,
        'host_documents': host_documents,
        'embeddings_model_name': embeddings.model_name,
        'texts': texts,
        'faiss_index': vectorStore.serialize_to_bytes()  # Serialize the FAISS index
    }

    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(metadata, f)

def get_embeddings(store_name:str, path:str, embeddings:HuggingFaceEmbeddings, **kwargs):
    embeddings_path = f"{path}/faiss_{store_name}.pkl"
    if not os.path.exists(embeddings_path):
        if not kwargs['host_documents']:
            texts = ['']
            faiss_index = FAISS.from_texts(texts, embeddings)
        else:
            docs = kwargs['docs']
            texts = [x.page_content for x in docs]
            faiss_index = FAISS.from_documents(docs, embeddings)

        # Create a dictionary containing the metadata    
        metadata = {
            'store_name': store_name,
            'host_documents': kwargs['host_documents'],
            'embeddings_model_name': embeddings.model_name,
            'texts': texts,
            'faiss_index': faiss_index.serialize_to_bytes()  # Serialize the FAISS index
        }

        # Guardar metadata
        with open(embeddings_path, "wb") as f:
            pickle.dump(metadata, f)
    
    with open(embeddings_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Deserialize the FAISS index 
    faiss_index = FAISS.deserialize_from_bytes(metadata['faiss_index'], embeddings)
    metadata['faiss_index'] = faiss_index

    return metadata

def convert_language_variable(language_variable):
    # Define el patrón de búsqueda utilizando expresiones regulares
    pattern = r'^(en)$|([a-z]{2})[-_]?([a-z]{2})?$'

    # Intenta hacer el reemplazo
    match = re.match(pattern, language_variable)

    value = None
    if match:
        # Si hay coincidencia con el patrón, toma la parte correspondiente del idioma
        if match.group(1):
            value =  'en_us'
        elif match.group(2):
            value = match.group(2)
    else:
        value = language_variable

    return value
