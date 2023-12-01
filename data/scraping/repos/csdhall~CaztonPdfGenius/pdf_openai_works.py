from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the OPENAI_API_KEY environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# location of the pdf file/files. 
reader = PdfReader('docs/RestrictAct.pdf')

reader

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# raw_text

raw_text[:100]

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

len(texts)

texts[0]

texts[1]

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

docsearch

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "who are the authors of the article?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)

# Start an infinite loop to continuously ask questions
while True:
    # Prompt the user to enter a question
    query = input("Enter your question (or type 'exit' to quit): ")
    
    # Check if the user wants to exit the loop
    if query.lower() == 'exit':
        break

    # Perform similarity search using the query
    docs = docsearch.similarity_search(query)
    
    # Run the question-answering chain
    response = chain.run(input_documents=docs, question=query)
    
    # Print the response
    print(response)

# Exit message
print("Exiting the question-answering loop.")

