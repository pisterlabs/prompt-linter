import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#set env variable for openai
os.environ["OPENAI_API_KEY"] = "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"


#instance of LLM with high temp for more randomness

LLM1 = OpenAI(temperature=0.9)


#1 testing instance ---

#q1="What is a good name for an aggressive pitbull dog"

#print(LLM1(q1))


#2 prompts  ---

prompt1= PromptTemplate( input_variables = ["animal"], template="What is a good name for a {animal} ",)

#print(prompt1.format(animal="dog"))



#3 building the LLM ---

firstchain = LLMChain(llm=LLM1,prompt=prompt1)

inp1=input("Enter animal:")

print(firstchain.run(inp1))


#4 SerpAPI- for getting google search results .









