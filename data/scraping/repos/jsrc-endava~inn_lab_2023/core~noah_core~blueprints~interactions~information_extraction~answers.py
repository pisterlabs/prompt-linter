import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Milvus


load_dotenv('.env')


def answer(query):
  docsearch = Milvus(
    embedding_function=OpenAIEmbeddings(),
    collection_name="messages",
    connection_args={
      "uri": os.environ['ZILLIZ_ENDPOINT'],
      "token": os.environ['ZILLIZ_TOKEN'],
    },
  )

  qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

  return {
      "answer": qa.run(query)
  }
