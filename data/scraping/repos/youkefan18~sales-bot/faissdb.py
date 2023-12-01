import os
import sys
from abc import ABC, abstractmethod

import faiss
import numpy as np
from langchain.docstore import InMemoryDocstore
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain.vectorstores import FAISS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.chinese_embedding import ChineseEmbedding

from vectordbs.vectordb import VectorDb


def score_normalizer(val: float) -> float:
    """_Issue with similarity score threshold_
    Below is not working for HuggingFaceEmbeddings since the similarity score is not scaled to [0,1]
    See [issue](https://github.com/langchain-ai/langchain/issues/4086)
        retriever=FaissDb().db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8}
            )
    Solve this by init FAISS with relevance_score_fn = score_normalizer.
    Or pass above param to FAISS.from_documents().
        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val))
    """
    return 1 - 1 / (1 + np.exp(val))

class FaissDb(VectorDb):
    """_summary_

    The filename is not allowed to be named as faiss.py, 
    otherwise it will pop up strange error like 'module 'faiss' has no attribute 'IndexFlatL2''
    See [issue](https://github.com/facebookresearch/faiss/issues/1195)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    #override
    def _initDb(self, dbfile: str, embedding: Embeddings, rebuild: bool) -> VectorStore:
        _db: FAISS = None
        if not os.path.exists(dbfile.replace(".txt", ".db")) or rebuild:
            try:
                with open(dbfile, 'r', encoding='utf-8-sig') as f:
                    docs = f.read()
                docs = self._transformer.create_documents([docs])
                _db = FAISS.from_documents(docs, embedding, relevance_score_fn=score_normalizer)
                _db.save_local(dbfile.replace(".txt", ".db"))
                return _db
            except FileNotFoundError as e:
                print(e)
            except Exception as e:
                print(e)
        else:
            _db = FAISS.load_local(dbfile.replace(".txt", ".db"), embedding,  relevance_score_fn=score_normalizer)
        return _db
    
    #override
    def createMemory(self) -> VectorStoreRetrieverMemory:
        embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = self._embedding
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        return memory

if __name__ == "__main__":
    v = FaissDb()
    retriever = v.db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}
    )
    query = "你们价格怎么这么贵，是不是在坑人？"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.page_content + "\n")