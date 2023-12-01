from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chat_models import ChatOpenAI

import tiktoken
# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(".env", override=True)

# Access the OPENAI_API_KEY environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_type = os.environ.get("OPENAI_API_TYPE")
openai_api_base = os.environ.get("OPENAI_API_BASE")
openai_api_version = os.environ.get("OPENAI_API_VERSION")

# print(openai_api_key)
# print(openai_api_type)
# print(openai_api_base)
# print(openai_api_version)

# location of the pdf file/files. 
file_name= 'GPT-FATHOM'
file_path = f'docs/{file_name}.pdf'

reader = PdfReader(file_path)

reader

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# raw_text

raw_text[:100]
raw_text.replace("<|endoftext|>", "endoftext")

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 30000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

len(texts)

texts[0]

texts[1]

texts = [item.replace("<|endoftext|>", "endoftext") for item in texts]

# Download embeddings from OpenAI
embeddings_model = "CaztonEmbedAda2"
tokenizer = tiktoken.get_encoding("cl100k_base")

embeddings = OpenAIEmbeddings(deployment=embeddings_model,
                              openai_api_base=openai_api_base,
                              openai_api_version=openai_api_version,
                              openai_api_key=openai_api_key,
                              openai_api_type=openai_api_type,
                              chunk_size=1)

import pickle  
  
# NOTE: DO THIS FIRST. Uncomment and run.  
# with open(f'Data/{file_name}.pkl', 'wb') as f:  
#     pickle.dump(embeddings, f)  

# exit()
# import pickle  

file = f'Data/{file_name}.pkl'
with open(file, 'rb') as f:  
    print(f"Reading.........{file}")
    embeddings = pickle.load(f)  

docsearch = FAISS.from_texts(texts, embeddings)

docsearch

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# gpt3_model = "CaztonDavinci3"
gpt4_model = "CaztonGpt-4"

chain = load_qa_chain(ChatOpenAI(engine=gpt4_model, temperature=0.3, max_tokens=3000), chain_type="stuff")
# query = "who are the authors of the article?"
# docs = docsearch.similarity_search(query)
# chain.run(input_documents=docs, question=query)

# Start an infinite loop to continuously ask questions

# NOTE: Uncomment WHILE loop to test
# while True:
#     # Prompt the user to enter a question
#     query = input("Enter your question (or type 'exit' to quit): ")
    
#     # Check if the user wants to exit the loop
#     if query.lower() == 'exit':
#         break

#     # Perform similarity search using the query
#     docs = docsearch.similarity_search(query)
    
#     # Run the question-answering chain
#     response = chain.run(input_documents=docs, question=query)
    
#     # Print the response
#     print(response)

# # Exit message
# print("Exiting the question-answering loop.")

from fastapi import FastAPI, HTTPException, Depends  
from fastapi.middleware.cors import CORSMiddleware  
import uvicorn  
  
app = FastAPI()  
  
# Add CORS middleware  
origins = [  
    "http://localhost",  
    "http://localhost:5000",  
    "http://127.0.0.1",  
    "http://127.0.0.1:5000",  
]  
  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  
  
# ... (your existing code)  


from pydantic import BaseModel  
  
class QuestionRequest(BaseModel):  
    key: str  


@app.post("/qa")  
async def ask_question(request: QuestionRequest):  
    query = request.key  
    print("***********")  
    print(query)  
    if not query:  
        raise HTTPException(status_code=400, detail="Query must not be empty")  
  
    docs = docsearch.similarity_search(query)  
    response = chain.run(input_documents=docs, question=query)  
    print(response)
 
    return {"response": response} 

 


if __name__ == "__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8080)  


