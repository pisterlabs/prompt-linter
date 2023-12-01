import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SUMMARIZING DOCUMENT
    # load_summarize_chain

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

llm = OpenAI(temperature=0.9)

with open("Sample.txt") as f:
    data = f.read()

text_splitter = CharacterTextSplitter() # Split the text
texts = text_splitter.split_text(data)

docs = [Document(page_content=t) for t in texts] # Create multiple documents
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(docs)

# HHTP REQUEST
    # LLMRequestsChain

from langchain.chains import LLMRequestsChain, LLMChain

template = """
Extract the answer to the question '{query}' or say "not found" if the information is not available.
{requests_result}
"""

PROMPT = PromptTemplate(input_variables=["query", "requests_result"], template=template)

llm=OpenAI()

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=PROMPT))

question = "What is the capital of india?"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+"),
}

chain(inputs)