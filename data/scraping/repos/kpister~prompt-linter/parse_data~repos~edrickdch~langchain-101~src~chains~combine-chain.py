from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.9)
first_prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase \
        for the following company: {company_name}",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

catchphrase = overall_chain.run("colorful socks")
print(catchphrase)
