from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate



prompt = PromptTemplate.from_template("what is the name of the e commerce store from where you can buy {product}?")

llm = OpenAI(temperature=0.3)
chain = LLMChain(llm=llm, prompt=prompt)
#product = "phone"
#result = chain.run("phone")
#print(result)

#LLM to get comma separated names of products from an e commerce store name
prompt = PromptTemplate.from_template("what are the names of the products available on {store}?")
llm = OpenAI(temperature=0.3)
chain2 = LLMChain(llm=llm, prompt=prompt)
#store = "amazon"
#result = chain.run(store)
#print(result)


# create an overall chain from simple sequential chains
chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)
output = chain.run("candles")
print(output)
