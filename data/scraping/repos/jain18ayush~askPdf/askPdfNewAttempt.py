from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI

import sys 
print(sys.executable)


llm = OpenAI(temperature=0)
persist_directory = './db'

# loads a pdf into a document object
loader = DirectoryLoader('/Users/ayushjain/OneSpace/Untitled/pdf/', glob="./*.pdf", loader_cls=PyPDFLoader)
# loader = PyPDFLoader("/Users/ayushjain/Downloads/case 1.pdf")
data = loader.load()

# the other option is the character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, length_function=len,)
# the splits are fed into the embedder and then fed into the vector store
all_splits = text_splitter.split_documents(data)

# # creates the vector store using the OPENAI Embedding
# vectorstore = Chroma.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)

# vectorstore.persist()

# #how to retrieve data at a later point 
# vectorstore = None 
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
vectorstore.persist()

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
)

chat_history = []
query = "What was the verdict of the gaskins case? Was it vacated?"
result = chain({"question": query, "chat_history": chat_history})
result = chain({"question": "Why was it vacated?", "chat_history": chat_history})
result = chain({"question": "What was the obligation?", "chat_history": chat_history})

print(result["answer"])
