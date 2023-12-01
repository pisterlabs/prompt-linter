import openai
import secret
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# print(secret.API_KEY)

key =  secret.API_KEY

def capital_finder(place):
    llm = OpenAI(openai_api_key=key, temperature=.5)
    prompt_temp = PromptTemplate.from_template("What is the capital of {place}?")
    capital_text = LLMChain(llm=llm, prompt=prompt_temp)
    output = capital_text.run(place)
    return output

# print(capital_finder("India"))