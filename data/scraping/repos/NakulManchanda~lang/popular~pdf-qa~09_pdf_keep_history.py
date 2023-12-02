# https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
# https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb

# using chroma db to store embedding, and use similarity search to retrieve most relevant chunk of text which can answer the question
# helps with large pdfs
import logging
import os
from dotenv import load_dotenv
load_dotenv()

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
SERPAPI_API_KEY=os.environ["SERPAPI_API_KEY"]
PDF_ROOT_DIR=os.environ["PDF_ROOT_DIR"]

from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# load document
loader = PyPDFLoader(f"{PDF_ROOT_DIR}/sample.pdf")
documents = loader.load()


from langchain.chains import RetrievalQA
# https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
from langchain.text_splitter import CharacterTextSplitter
# https://python.langchain.com/en/latest/reference/modules/embeddings.html
from langchain.embeddings import OpenAIEmbeddings
# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
chat_history = []

# take user input python in while loop
while True:
    query = input("Ask me a question: ")
    response = qa({"question": query, "chat_history": chat_history})
    print(response["answer"])
    chat_history.append((query, response["answer"]))