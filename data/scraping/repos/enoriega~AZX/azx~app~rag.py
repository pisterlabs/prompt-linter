import csv
import itertools as it
import os

import openai
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI


from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(path: str):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        ret = [Document(page_content=r['Text content']) for r in reader]

    return ret

def build_rag_chain(path:str, llm):
    documents = load_data(path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    # rag_prompt = hub.pull("rlm/rag-prompt")
    rag_prompt = PromptTemplate.from_template("You are a public health advocate. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Enumerate the actionable items as clear and consise bullets in a list. Write a follow up sentence that briefly elaborates the main sentence in each item.\nQuestion: {question} \nContext: {context} \nAnswer:")
    retriever = vectorstore.as_retriever()
    # llm = ChatOpenAI(temperature=0, model='gpt-4-1106-preview')
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm

    return rag_chain


if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model='gpt-4-1106-preview')
    rag_chain = build_rag_chain("azx_data.tsv", llm)

    x = rag_chain.invoke("What should I do in the presence of poor air quality?")

    print(x)

