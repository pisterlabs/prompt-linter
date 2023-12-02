"""
一次查询可能得不到满意的结果，人工可以通过手动定位问题并调整来解决，但是会很枯燥。

MultiQueryRetriever 是为了自动化这个枯燥的过程，他自动通过 LLM 生成多个查询，作为用户输入查询的代表，去做查询。

对于每个查询，它都会检索一组相关文档，并采用所有查询之间的唯一并集来获取更大的一组潜在相关文档。
通过对同一问题生成多个视角，MultiQueryRetriever 或许能够克服基于距离的检索的一些限制，并获得更丰富的结果集。
"""
import env

# Build a sample vectorDB
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load blog post
loader = WebBaseLoader("https://baijiahao.baidu.com/s?id=1773015278425002420")
data = loader.load()

print("web loaded data:", data)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(data)

print("web splits data:", splits)

# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

# 指定 LLM 生成查询
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
question="What are the approaches to Task Decomposition?"
llm = ChatOpenAI(max_tokens=8000, model_name="gpt-3.5-turbo-16k-0613", temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)

# Set logging for the queries
import logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print("unique_docs len:", len(unique_docs))
print("unique_docs:", unique_docs[:3])

"""
你也可以自己提供 prompt
"""

from typing import List
from langchain import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


# 将输出分割成一个列表
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
llm = ChatOpenAI(temperature=0)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Other inputs
question = "简述一下这篇文章?"

# "lines" is the key (attribute name) of the parsed output
retriever = MultiQueryRetriever(retriever=vectordb.as_retriever(), llm_chain=llm_chain,parser_key="lines")

# Results
unique_docs = retriever.get_relevant_documents(query="为什么没有能正确防备危险的到来?")
len(unique_docs)