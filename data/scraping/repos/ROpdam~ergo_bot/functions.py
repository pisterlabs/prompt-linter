import os
from typing import Union

import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

prompt_template = """Use the following pieces of context to answer
the question at the end. If you don't know the answer, just
say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer using bullet points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
k = 5

load_dotenv()
db_location = os.getenv("DB_LOCATION")

# OpenAI embeddings
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = OpenAIEmbeddings()

vectordb = FAISS.load_local(db_location, embedding_function)
retriever = vectordb.as_retriever(search_kwargs={"k": k})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=False,
    chain_type_kwargs=chain_type_kwargs,
)


def get_article_links(docs: list) -> Union[list, list]:
    """"""
    article_names = "Ergolog Articles"
    content = ""
    for idx, doc in enumerate(docs):
        link_name = f" Article {idx+1} "
        content += (
            "\n"
            + "{}\n[{}]({})".format(
                doc.metadata["article_title"], link_name, doc.metadata["article_link"]
            )
            + "\n"
        )

    return [cl.Text(name=article_names, content=content)], article_names


def format_response(llm_response: dict) -> dict:
    """"""
    article_links, article_names = get_article_links(llm_response["source_documents"])

    return {
        "answer": llm_response["result"],
        "article_links": article_links,
        "article_names": article_names,
    }
