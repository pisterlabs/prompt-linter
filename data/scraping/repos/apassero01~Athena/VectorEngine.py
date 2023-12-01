from dotenv import load_dotenv
from django.db import transaction
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
import os

class VectorEngine(): 
    def __init__(self, vectorCollectionName = "KBitem" ): 
        load_dotenv()
        self.embeddings = OpenAIEmbeddings()

        self.vectorCollectionName = vectorCollectionName

        self.CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
            host=os.environ.get("PGVECTOR_HOST", "localhost"),
            port=int(os.environ.get("PGVECTOR_PORT", "5432")),
            database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
            user=os.environ.get("PGVECTOR_USER", "postgres"),
            password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
        )

        self.vectorDatabase = PGVector(
            collection_name=self.vectorCollectionName,
            connection_string=self.CONNECTION_STRING,
            embedding_function=self.embeddings,
        )


    
    def TextToDocs(self, text, kbItemID, userID): 
        metadata = {"kbItemID": kbItemID,"userID": userID}
        document = Document(page_content=text, metadata=metadata)
        return document
    

    def storeVector(self, documents, chunk_size = 100): 
        textSplitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=0)
        documents = textSplitter.split_documents(documents=[documents])
        longDocuments = [d for d in documents if len(d.page_content) > 50]

        if len(longDocuments) > 1: 
            documents = longDocuments

        with transaction.atomic():   
            vectorIDs = self.vectorDatabase.add_documents(documents=documents)
        return vectorIDs
    
    def getMatchedDocs(self, queryString,userID): 
        matched_docs = self.vectorDatabase.similarity_search_with_score(queryString, k = 50,filter={"userID":userID})
        kbItemIDs = {}
        for doc in matched_docs: 
            kbItemID = doc[0].metadata['kbItemID']
            if kbItemID not in kbItemIDs: 
                kbItemIDs[kbItemID] = doc[1]
        
        return kbItemIDs
