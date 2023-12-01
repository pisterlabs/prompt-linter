from langchain.evaluation.loading import load_dataset
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AwaDB
import openai
import awadb
from langchain.vectorstores import FAISS
import pandas as pd
import os


dataset = load_dataset("question-answering-state-of-the-union")
loader = TextLoader('./data/state_of_the_union.txt', encoding='utf8')
data = loader.load()
# Create split file
text_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=5)
split_docs = text_splitter.split_documents(data)
docs = []
for text in split_docs:
    docs.append(str(text))

'''
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)
db.save_local("faiss_index_2")
'''

openai.api_key = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()
#doc_result = embeddings.embed_documents(docs)
db = AwaDB.from_documents(docs)

db.similarity_search()



'''
embeddings_path = 'embedded.csv'
df = pd.DataFrame({"text": docs, "embedding": doc_result})
df.to_csv(embeddings_path, index=False)
'''
'''
embeddings_path = 'output/embedded.csv'
embeddings = []
def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = text, model=model)['data'][0]['embedding']
for text in split_docs:
    embeddings.append(get_embedding(text, model='text-embedding-ada-002'))
df = pd.DataFrame({"text": split_docs, "embedding": embeddings})
df.to_csv(embeddings_path, index=False)
'''