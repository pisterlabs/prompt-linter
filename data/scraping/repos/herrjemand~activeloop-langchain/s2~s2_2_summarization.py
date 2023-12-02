from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

llm = OpenAI(model="text-davinci-003", temperature=0)

summarize_chain = load_summarize_chain(llm=llm)

document_loader = PyPDFLoader(file_path="/Users/yuriy/Downloads/inz1242-signed-fixed.pdf")
document = document_loader.load()

summary = summarize_chain.run(document)

print(summary)