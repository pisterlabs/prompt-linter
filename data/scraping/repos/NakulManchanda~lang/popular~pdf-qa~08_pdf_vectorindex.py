# https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
# https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb
# using VectorstoreIndexCreator to index the embeddings, and use similarity search to retrieve most relevant chunk of text which can answer the question
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

# load document
loader = PyPDFLoader(f"{PDF_ROOT_DIR}/sample.pdf")
documents = loader.load()

# https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
from langchain.text_splitter import CharacterTextSplitter
# https://python.langchain.com/en/latest/reference/modules/embeddings.html
from langchain.embeddings import OpenAIEmbeddings
# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator


index = VectorstoreIndexCreator(
    # split the documents into chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    # select which embeddings we want to use
    embedding=OpenAIEmbeddings(),
    # use Chroma as the vectorestore to index and search embeddings
    vectorstore_cls=Chroma
).from_loaders([loader])

# take user input python in while loop
while True:
    query = input("Ask me a question: ")
    print(index.query(llm=OpenAI(), question=query, chain_type="stuff"))