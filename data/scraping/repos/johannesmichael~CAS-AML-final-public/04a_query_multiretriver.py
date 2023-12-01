from typing import List
from langchain import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
# Set logging for the queries
import logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
import argparse

from dotenv import load_dotenv
import os

load_dotenv()

from scripts.constants import CHROMA_SETTINGS
persist_directory = os.environ.get('PERSIST_DIRECTORY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')



# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)
    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()
    
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions seperated by newlines.
    Original question: {question}""",
)





# create the top-level parser
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--collection_name", help="Name of the collection to create/use")
    parser.add_argument("--max_tokens", help="max tokens to generate", default=1000)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collection_name = args.collection_name
    max_tokens = args.max_tokens

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=max_tokens)
    # Chain
    llm_chain = LLMChain(llm=llm,prompt=QUERY_PROMPT,output_parser=output_parser)

    embeddings = OpenAIEmbeddings()

    db = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)


    # Run
    retriever = MultiQueryRetriever(retriever=db.as_retriever(), 
                                    llm_chain=llm_chain,
                                    parser_key="lines") # "lines" is the key (attribute name) of the parsed output


    while True:
        query = input("> ")
        answer = retriever.get_relevant_documents(query=query)
        print(answer)

# python 04a_query_multiretriver.py --collection_name openai_ada_1000cs --max_tokens 1000