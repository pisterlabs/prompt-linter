from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

loader = DirectoryLoader('./md_docs', glob='**/*.mdx', loader_cls=TextLoader, show_progress=True)
docs = loader.load()

md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = md_splitter.split_documents(docs)

print(splits[0])

persist_dir = './md_vectors'

# delete the persist_dir if it exists
import shutil
try:
    shutil.rmtree(persist_dir)
except:
    pass


vectordb = Chroma.from_documents(persist_directory=persist_dir, embedding=OpenAIEmbeddings(), documents=splits)

vectordb.persist()

query = "How to summarize the average of a field"

docs = vectordb.similarity_search(query, k=5)

print(len(docs[1].page_content))



