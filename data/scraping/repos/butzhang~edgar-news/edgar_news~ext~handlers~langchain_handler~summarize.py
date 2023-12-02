from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


def get_summarization(text) -> str:
    llm = OpenAI(temperature=0)

    text_splitter = CharacterTextSplitter()

    # texts = text_splitter.split_text(text)
    docs = [Document(page_content=text)]
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(docs)
