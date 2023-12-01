from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone 
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

my_loader = DirectoryLoader('./data_output_filtered_content', glob='**/*.txt')
documents = my_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# initialize pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=os.environ['PINECONE_ENV']  # next to api key in console
)

docsearch = Pinecone.from_documents(docs, embeddings, index_name=os.environ['PINECONE_INDEX_NAME'])

query = "write me langchain code to build my hugging face model"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)

# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)