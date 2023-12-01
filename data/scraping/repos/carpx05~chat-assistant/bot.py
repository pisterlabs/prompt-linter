import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv   #for python-dotenv method
load_dotenv()
# from langchain.document_loaders import JSONLoader
# import json
# from pathlib import Path
# from pprint import pprint

os.environ.get["OPENAI_API_KEY"]

# from langchain.docstore.document import Document

# from langchain.document_loaders import TextLoader
# # Load text data from a file using TextLoader
# loader = TextLoader("data.txt")
# document = loader.load()

from langchain.document_loaders import DirectoryLoader

directory = 'C:/Users/ayush/Documents/Team Envision/data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)




def split_docs(documents, chunk_size = 2000, chunk_overlap= 0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n"]
    )
    docs = text_splitter.split_documents(documents)
    return docs
  
docs = split_docs(documents)
print("Number of questions: ", len(docs))
# for i in range(0,len(docs)):
#     print(docs[i].page_content)
#     print("\n")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# pinecone credentials
pinecone.init(
    api_key="Pinecone_API_Key",
    environment="Pinecone_Environment"
)

index_name = "eva"
print(index_name, " Stats") 
if index_name not in pinecone.list_indexes(): #creates an index if not present
    pinecone.create_index(index_name, dimension=1536, metric='cosine')

print(pinecone.Index("eva").describe_index_stats()) #gives index stats

goahead = False
while goahead == False:
  user_input = input("""If you want to add data to the vector db, press "yes" or "1", else (this option is to retrieve existing data) press "no" or "0": """)
  if user_input.lower() in ['yes', '1']:
    index = Pinecone.from_documents(docs, embeddings, index_name = index_name) # concatenates data from txt file to vector stores
    goahead = True
  elif user_input.lower() in ['no', '0', '']:
    index = Pinecone.from_existing_index(index_name, embeddings) # retrieves existing data
    goahead = True
  else:
    print("Invalid input. Please choose 'yes' or '1' to add data, or 'no' or '0' to retrieve existing data.")


def get_similar_docs(query,k=4,score=False): 
  if score:
    similar_docs= index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k) #both functions use cosine similarity
  return similar_docs

print("similar_docs working...")

model_name = "gpt-3.5-turbo" # Enter your llm model
llm = OpenAI(model_name=model_name, temperature = 0) # adjust your temp [0-1] (A higher temperature value typically makes the output more diverse and creative but might also increase its likelihood of straying from the context)

chain = load_qa_chain(llm, chain_type= "stuff")

def get_answer(query): # response function
  similar_docs_with_score = get_similar_docs(query, score = True)

  similar_docs = get_similar_docs(query)
  question = query + "Please don't make up answers"
  answer = chain.run(input_documents=similar_docs, question = question)
  return answer

query = "who is the organiser of agritech?"
answer = get_answer(query)
print(answer)

print("get_answer working...")