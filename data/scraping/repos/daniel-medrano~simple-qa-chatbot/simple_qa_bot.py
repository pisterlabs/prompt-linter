from prompt_templates import *

from langchain import PromptTemplate, OpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os

from dotenv import load_dotenv

load_dotenv()

def get_plain_answer(question):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    pinecone.init(
        api_key=pinecone_api_key,
        environment="asia-southeast1-gcp-free"
    )

    index = pinecone.Index("chatbot")

    model_name = "text-embedding-ada-002"

    embeddings = OpenAIEmbeddings(
        model = model_name,
        openai_api_key=openai_api_key,
    )

    vectorstore = Pinecone(
        index, 
        embeddings, 
        "text"
    )

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )

    return qa.run(question)

def get_formatted_answer(plain_answer):
    llm = OpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    prompt_template = PromptTemplate(input_variables=["plain_answer"], template=GENERATE_FORMATTED_ANSWER)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain.run({ "plain_answer": plain_answer })

def generate_answer(question):
    print("Question: " + question)
    plain_answer = get_plain_answer(question)
    print("Plain Answer: " + plain_answer)
    formatted_answer = get_formatted_answer(plain_answer)
    return formatted_answer

while True:
    question = input("What is your question? ")
    print(generate_answer(question))
