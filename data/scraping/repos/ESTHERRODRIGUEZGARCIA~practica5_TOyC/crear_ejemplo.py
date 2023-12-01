from parte2 import *
from pruebaia import *
from langchain.llms import OpenAI
from langchain.chains import GraphQAChain



chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)
chain.run(question)