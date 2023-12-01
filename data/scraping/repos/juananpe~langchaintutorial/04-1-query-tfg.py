# -*- coding: utf-8 -*-

from dotenv import load_dotenv

# Load the API key from an environment variable or file
load_dotenv()

from langchain.document_loaders import PyPDFLoader
url = "https://addi.ehu.es/bitstream/handle/10810/50524/TFG_OihaneAlbizuriSilguero.pdf"
loader = PyPDFLoader(url)
documents = loader.load()
print("Document loaded. Start running the chain...")

# Create LLM
from langchain.chat_models import ChatAnthropic
llm = ChatAnthropic()

# Create QA chain
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm=llm, chain_type="stuff")

query = "Brevemente, ¿de qué trata el proyecto? ¿quién lo ha realizado? ¿cuál es su objetivo?"
answer = chain.run(input_documents=documents, question=query)
print(answer)


'''
# Run the chain
query = "¿Cuáles son los riesgos del proyecto?"
answer = chain.run(input_documents=documents, question=query)
print(answer)
'''
