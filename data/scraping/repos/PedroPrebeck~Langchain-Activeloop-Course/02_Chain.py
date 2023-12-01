from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.9)

template = "What is a good name for a company that makes {product} products?"

prompt = PromptTemplate(input_variables=["product"], template=template)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("eco-friendly water bottles"))
