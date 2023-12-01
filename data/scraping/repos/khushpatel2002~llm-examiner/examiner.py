from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.llms import OpenAI

from dotenv import dotenv_values
import os

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
persist_directory = 'db'

embedding = OpenAIEmbeddings()
vectordb2 = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

 
llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


qa_chain.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template = """
Use the following pieces of context to Examine the users answers for the provided question.
If you answer is not correct, just say that wrong answer or the percentage of correctness.
----------------
{context}"""

def process_llm_response(llm_response):
    return (llm_response['result'])
def examine_answer(QA_PAIR):
  query = QA_PAIR
  llm_response = qa_chain(query)
  result = process_llm_response(llm_response)
  return result


# usage
# question = "What is the purpose of fasting?"
# answer = "The purpose of fasting is to exercise self-restraint for the sake of Allah and to develop God-consciousness."

# QA_PAIR = f"Q. {question} A. {answer}"
# print(examine_answer(QA_PAIR))