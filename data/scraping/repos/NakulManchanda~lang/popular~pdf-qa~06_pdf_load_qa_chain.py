# https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
# https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb

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
chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")

# take user input python in while loop
while True:
    query = input("Ask me a question: ")
    print(chain.run(input_documents=documents, question=query))