from langchain.vectorstores import Chroma
from langchain import embeddings, text_splitter

from typing import Optional

import time

#  uncomment for linux
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



def store_db(texts: text_splitter, embeddings: embeddings, persist_directory: Optional[str], persist: Optional[bool]):
    '''
        function to store the database

        parameters:
            texts: list of text objects
            embeddings: embeddings to apply
            persist_directory: directory to save the database
            persist: whether to persist the database or not

        returns:
            vector_store: vector store object
    '''

    vector_db = None
    def store(text):

        #if persist is true and persist_directory is not None or persist is false and persist_directory is not None
        if (persist and persist_directory is not None) or (not persist and persist_directory is not None):
            #store texts into directory
            print(f"Storing database in directory '{persist_directory}'...")

            vector_db = vector_store.from_documents(
                                            documents=text, 
                                            embedding=embeddings,
                                            persist_directory=persist_directory
                                        )
            
        #else if persist is true and persist_directory is None
        elif persist and persist_directory is None:
            #store texts into memory
            raise Exception("Persist directory must be specified to persist the database")
        #else if persist is false and persist_directory is None
        elif not persist and persist_directory is None:
            print("Storing database in memory...")
            #store texts into memory
            vector_db = vector_store.from_documents(documents=texts, 
                                    embedding=embeddings)
            
        return vector_db


    print('total docs to be stored: ', len(texts))

    #load vector store
    vector_store = Chroma()
    
    # get chroma db max batch size
    from chromadb import Client
    chromadb_client = Client()
    chroma_batch_size = chromadb_client.max_batch_size

    if chroma_batch_size > 5000:
        chroma_batch_size = 5000

    #if the number of texts is greater than the chroma db max batch size
    if len(texts) > chroma_batch_size:
        #split texts into batches
        texts = [texts[i:i+chroma_batch_size] for i in range(0, len(texts), chroma_batch_size)]
        print('total batches: ', len(texts))

        #for each batch in texts
        # store batch
        # keep count of the number of batches
        # after every 3rd batch sleep for 1.5 minutes
        # this is to prevent the chroma db from crashing
        for i, batch in enumerate(texts):
            #store batch
            vector_db = store(batch)
         #   time.sleep(1.5)
            #if i % 3 == 0:
            print('sleeping for 90 seconds...')
            time.sleep(90)
    else:
        #store texts
        vector_db = store(texts)

    
    if vector_db is None:
        raise Exception('Database could not be stored')
    
    return vector_db
