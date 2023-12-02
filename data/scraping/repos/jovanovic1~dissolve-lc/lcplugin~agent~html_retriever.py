from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

class HtmlRetrieval:
    def __init__(self, query: str, url: str):
        self.query = query
        self.url = url

    def retrieve_html(self):
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        url = ''.join(filter(str.isalnum, self.url))
        persist_directory = 'vectordb'
        embeddings = OpenAIEmbeddings()

        # print("url is: ", url)

        db = Chroma(persist_directory=persist_directory, collection_name=url, embedding_function=embeddings)
        
        docs = db.similarity_search(self.query)
        # print(docs[0])
        return docs[0]
# htmlr = HtmlRetrieval("Click on the link to navigate to bluetooth headsets page", "http://www.logitech.com/en-in/products/headsets.html")
# htmlr.retrieve_html()