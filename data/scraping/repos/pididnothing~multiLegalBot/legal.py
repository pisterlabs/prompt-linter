import os
from langchain.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = "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"

pdf_loader = PyPDFLoader('./docs/sample.pdf')
documents = pdf_loader.load()

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# we are specifying that OpenAI is the LLM that we want to use in our chain
chain = load_qa_chain(llm=OpenAI(), verbose=True)
query = 'What is the document about?'
response = chain.run(input_documents=documents, question=query)
print(response) 