# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
import pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os


from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("The 17 immutable laws In Implant Dentistry.pdf")

## Other options for loaders 
# loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

data = loader.load()

# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[30].page_content)} characters in your document')

# Note: If you're using PyPDFLoader then we'll be splitting for the 2nd time.
# This is optional, test out on your own data.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')





# Check to see if there is an environment variable with you API keys, if not, use what you put below
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '4d940a59-e43b-4147-9dfe-5362b5ae7f80')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-east-1-aws') # You may need to switch with your env


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "periobot" # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

query = "What are ethe clinical advantages of the platform switching?"
docs = docsearch.similarity_search(query)

# Here's an example of the first document that was returned
print(docs[0].page_content[:450])

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is the socket shield technique?"
docs = docsearch.similarity_search(query)

chain.run(input_documents=docs, question=query)

# Modified to print the result:
print(chain.run(input_documents=docs, question=query))