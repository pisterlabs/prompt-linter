import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from config.setting import Setting
from config.pinecone_setting import index_name

Setting()


class PineconeVS:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.index = pinecone.Index(index_name)
        self.vectorstore = Pinecone(
            self.index, self.embeddings.embed_query, 'text')
