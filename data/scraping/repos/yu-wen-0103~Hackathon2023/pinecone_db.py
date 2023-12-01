from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
import pinecone

os.environ["PINECONE_ENV"] = "gcp-starter"
os.environ["PINECONE_API_KEY"] = "6f605f74-d8ae-45c0-bdb1-aaf67686082b"
os.environ["OPENAI_API_KEY"] = "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"

class Pinecone_DB:
    def __init__(self, index_name):
        self.index_name = index_name
        # Initialize the Pinecone client
        pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pin econe.io
                environment=os.getenv("PINECONE_ENV"),  # next to api key in console
                )
        # create an index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=1536)
        # create an OpenAIEmbeddings object
        self.embeddings = OpenAIEmbeddings()
        # store the pinecone index
        self.pinecone_index = pinecone.Index(self.index_name)
        # create an vectorstore object
        self.vector_store = Pinecone(self.pinecone_index, self.embeddings.embed_query, "text")
        
    def add_text_to_index(self, text):
        """
        Vector store input is a "list" of text
        """
        self.vector_store.add_texts([text])
    
    def add_list_text_to_index(self, text_list):
        # add text to the index
        self.vector_store.add_texts(text_list)
        
        
    def search_document(self, query, topN=9):
        docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)
        results = docsearch.similarity_search(query)
        query_len = len(results) if len(results) < topN else topN
        results = [results[i].page_content for i in range(query_len)]
        return results
    
    def _add_test_data(self):
        """
        test_data: list of text
        """
        test_data = ["學生證", "信用卡", "帽子", "水壺"]
        self.vector_store.add_texts(test_data)
    
if __name__ == "__main__":
    pinecone_db = Pinecone_DB("events")
    pinecone_db._add_test_data(["看電影", "玩桌遊", "打麻將", "去運動"])
    pinecone_db.add_text_to_index("看漫畫")
    pinecone_db.add_list_text_to_index(["看Netflix", "看電視"])
    results = pinecone_db.search_document("玩拉密", topN=9)
    print(results)
        