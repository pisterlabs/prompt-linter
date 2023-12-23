import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from dotenv import load_dotenv
from langchain import LLMChain

load_dotenv(dotenv_path='keys.env')

llm = OpenAI(model_name="text-davinci-003",temperature=0.2,openai_api_key=os.getenv("OPENAI_API_KEY"))



from langchain.schema import AIMessage,HumanMessage,SystemMessage

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)

template=""""
You are an expert data scientist, explain the concept of {concept} in a few lines
"""
prompt=PromptTemplate(template=template,input_variables=["concept"])
chain=LLMChain(prompt=prompt,llm=llm)


second_prompt=PromptTemplate(template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words.",input_variables=["ml_concept"])
chain_two=LLMChain(llm=llm,prompt=second_prompt)

from langchain.chains import SimpleSequentialChain
overall_chain=SimpleSequentialChain(chains=[chain,chain_two],verbose=True)
explanation=overall_chain.run("NLP")
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_spliter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)
texts=text_spliter.create_documents([explanation])
print(texts)
from langchain.embeddings import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model_name="ada")
query_result=embeddings.embed_query(texts[0].page_content)
import os
import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
index_name="langchain-quickstart"
search=Pinecone.from_documents(texts,embeddings,index_name=index_name)
query="What is magic about NLP"
result=search.similarity_search(query)

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

agent_executor=create_python_agent(
    llm=OpenAI(temperature=0,max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)
agent_executor.run("Find the roots (zeros) if the quadratic function 3* x**2 +2*x-1")

