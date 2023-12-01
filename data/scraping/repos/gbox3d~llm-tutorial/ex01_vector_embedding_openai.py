#%%
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI

import pinecone
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
)

#%%
print("Hello VectorStore!")
loader = TextLoader(
    "./mediumblog1.txt"
)
document = loader.load()

print("Document loaded!")
print(len(document))

# %%
print(document[0].metadata)

# %% text splitter
text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=0)
texts = text_splitter.split_documents(document)
print(len(texts))

# %%
print(texts[0].metadata)
print(texts[0].page_content)
# %%

index_name = os.getenv("PINECONE_INDEX_NAME")

index_config = {
    "metric": "cosine",
    "dimension": 1536
}

# Step 1: Delete the existing index
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)
    print(f"Deleted index: {index_name}")

#%%
# Step 2: Recreate the index
# You should configure the index according to your requirements
pinecone.create_index(index_name, **index_config)
print(f"Created index: {index_name}")

#%% embeddings
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_documents(texts, embeddings,index_name="medium-blog-embedding-index")
print('Embeddings created!')
# %%
