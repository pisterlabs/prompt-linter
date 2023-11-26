import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

import os

# Install the package
#!pip install cohere
# get a new token: https://dashboard.cohere.ai/

# from getpass import getpass
# COHERE_API_KEY = getpass()

from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY"))
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))


from langchain.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(cohere_api_key=os.environ.get("COHERE_API_KEY"))
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
