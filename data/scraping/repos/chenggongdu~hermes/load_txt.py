import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from config.setting import Setting
from config.pinecone_setting import index_name

Setting()
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../../data/education_faq.txt')

loader = TextLoader(file_path, "UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "都有哪些好的教育机构？"
docs = docsearch.similarity_search(query)
print(docs)