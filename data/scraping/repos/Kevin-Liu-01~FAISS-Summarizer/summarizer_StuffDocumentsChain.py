#Using StuffDocumentsChain
#When we use load_summarize_chain with chain_type="stuff", we will use the StuffDocumentsChain.
#The chain will take a list of documents, inserts them all into a prompt, and passes that prompt to an LLM:

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",openai_api_key=OPENAI_API_KEY)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

print(stuff_chain.run(docs))