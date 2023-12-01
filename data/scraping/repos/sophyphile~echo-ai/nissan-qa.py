# Pull data from Nissan Owner's Manual
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI

# The Document Loader for PDF files - load up PDF into documents
from langchain.document_loaders import PyPDFLoader

# The Vector Store we will use
from langchain.vectorstores import FAISS

# The LangChain chain used to fetch the documents i.e. vectors
from langchain.chains import RetrievalQA

# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

loader = PyPDFLoader("data/nissan-owner-manual.pdf")
docs = loader.load_and_split()
print (f"You have {len(docs)} document(s)")
print (f"You have {len(docs[0].page_content)} characters in the first document")

# Set up the embedding engine
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo DB. 
docDb = FAISS.from_documents(docs, embeddings)

# Respond to a query by returning relevant docs
docs = docDb.similarity_search_with_relevance_scores("How can I remove the head restraint?", 3)
# print (docs)

for doc in docs:
    print(str(doc[0].metadata["page"]) + ":", doc[0].page_content[:500])

# Respond to a query with a direct answer
# Create the retrieval engine
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docDb.as_retriever())

query = "What do MT and SRS stand for? Return their definitions in a dictionary, where the key is the acronym and the value is the definition"
print(qa.run(query))

# Pass in docs with a prompt template that includes a system prompt.
# Think about the ask in terms of the architecture and how acronyms are best extracted. Options: loops through doc chunks, or a tagged section of the docs, or...?