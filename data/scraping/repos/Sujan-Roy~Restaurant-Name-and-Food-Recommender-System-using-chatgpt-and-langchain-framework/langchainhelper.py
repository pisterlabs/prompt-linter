import os
from dotenv import load_dotenv
import openai
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

#this is the .env file
load_dotenv()
#this is the openai api key
my_key= os.getenv("OPENAI_API_KEY") 
#this is the llm model
llm= OpenAI(temperature=0,openai_api_key=my_key) 

#this function will generate the restaurant name and food items for the given country name
def generate_restaurant_name_and_foodItems(country_name):
    #chain 1
    prompt_name= PromptTemplate(
    input_variables=["country_name"],
    template="I want to open a restaurant for {country_name} food. Please suggest me name for this."
    )
    name_chain= LLMChain(llm=llm, prompt=prompt_name,output_key="restaurant_name")


    prompt_food_name= PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest me some food items for {restaurant_name} food. Return it as a numerical number list."
    )
    food_chain= LLMChain(llm=llm, prompt=prompt_food_name,output_key="food_item")

    chain= SequentialChain(chains=[name_chain,food_chain], 
                        input_variables=["country_name"],
                        output_variables=["restaurant_name","food_item"]
                        
                        )

    response= chain(country_name)
    return response

#this is the main function
if __name__ == "__main__":  
    response= generate_restaurant_name_and_foodItems("Thailand")
    print(response)