import glob
import json
import os

import pandas as pd
import chromadb

from langchain import FAISS
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from data_import_utils import extract_entries_from_file, adapt_metadata_chroma
import spacy
from utils import get_compute_device

load_dotenv()
nlp = spacy.load("en_core_web_lg")


# annotate entities in markdown using spacy
def annotate_entities_spacy(text):
    """
    Given a markdown text, annotate entities as links using spacy.
    :param text:
    :return:
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'start': ent.start_char,
            'end': ent.end_char,
            'label': ent.label_
        })
    return entities


def documents_from_entries(entries):
    """
    Given a list of entries, create a list of documents.
    :param entries:
    :return:
    """
    documents = []
    for entry in entries:
        documents.append(Document(page_content=entry['text'],
                                  metadata=adapt_metadata_chroma(entry['metadata'])))
    return documents


def chunks_from_entry(self, entries):
    # split entries into overlapping windows
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_docs = []
    for entry in entries:
        chunks = text_splitter.split_text(entry['text'])
        for chunk in chunks:
            doc = {
                'metadata': entry['metadata'],
                'text': chunk
            }
            all_docs.append(doc)

    return all_docs


def embed_index(doc_list, embed_fn, index_store):
    """Function takes in existing vector_store,
    new doc_list and embedding function that is
    initialized on appropriate model. Local or online.
    New embedding is merged with the existing index. If no
    index given a new one is created"""
    # check whether the doc_list is documents, or text
    try:
        faiss_db = FAISS.from_documents(doc_list,
                                        embed_fn)
    except Exception as e:
        faiss_db = FAISS.from_texts(doc_list,
                                    embed_fn)

    if os.path.exists(index_store):
        local_db = FAISS.load_local(index_store, embed_fn)
        # merging the new embedding with the existing index store
        local_db.merge_from(faiss_db)
        print("Merge completed")
        local_db.save_local(index_store)
        print("Updated index saved")
    else:
        faiss_db.save_local(folder_path=index_store)
        print("New store created...")


class JournalStore:
    def __init__(self, vector_dir, settings={}):

        args = {'device': get_compute_device()}
        text_embedding_model = settings.get('text_embedding_model', "BAAI/bge-large-en")
        event_embedding_model = settings.get('text_embedding_model', "BAAI/bge-large-en")
        vector_db_provider = settings.get('vector_db_provider', "faiss")

        self.text_embedding = SentenceTransformerEmbeddings(model_name=text_embedding_model,
                                                            model_kwargs=args)
        self.events_embedding = SentenceTransformerEmbeddings(model_name=event_embedding_model,
                                                              model_kwargs=args)
        self.vector_dir = vector_dir
        if vector_db_provider.lower() == "faiss":
            vectordb_path = os.path.join(self.vector_dir, 'text')
            eventsdb_path = os.path.join(self.vector_dir, 'events')

            if os.path.exists(vectordb_path):
                self.vectordb = FAISS.load_local(vectordb_path, self.text_embedding)

            if os.path.exists(vectordb_path):
                self.eventsdb = FAISS.load_local(eventsdb_path, self.events_embedding)

        elif vector_db_provider.lower() == "chroma":
            self.vectordb = Chroma(persist_directory=os.path.join(self.vector_dir, 'text'),
                                   embedding_function=self.text_embedding,
                                   collection_name="journal")
            self.eventsdb = Chroma(persist_directory=os.path.join(self.vector_dir, 'events'),
                                   embedding_function=self.events_embedding,
                                   collection_name="events")

    def import_journal_file(self, filename):
        entries = extract_entries_from_file(filename)
        # annotate entities
        for entry in entries:
            ents = annotate_entities_spacy(entry['text'])

        # summarise text into bullet points

        # give each entry a unique identifier

        documents = documents_from_entries(chunks)

        if not documents:
            print(f"File {filename} has no entries")
            return
        self.vectordb.add_documents(documents)
        self.vectordb.persist()

    def import_journal_files(self, filenames):
        for filename in filenames:
            self.import_journal_file(filename)

    def import_journal_dir(self, file_path):
        filenames = glob.glob(file_path)
        self.import_journal_files(filenames)

    def search_posts(self, query, metadata={}, limit=10, fetch_k=50):
        """
        Search posts in the vectorstore.
        :param query: text
        :param metadata: filter by a dict of boolean expressions
        :param limit: number of results to return
        :param fetch_k: number of results to fetch from the vectorstore
        :return: list of Document objects
        """

        if metadata:
            collection = self.vectordb._client.get_or_create_collection("journal")
            res = collection.query(query_texts=[query], where=metadata)
            zipped = zip(res['documents'][0], res['metadatas'][0])
            # Convert each tuple into a dictionary
            dicts = [dict(zip(['page_content', 'metadata'], tpl)) for tpl in zipped]

            documents = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in dicts]
            return documents
        else:
            return self.vectordb.search(query, "mmr", k=limit, fetch_k=50, where=metadata)
