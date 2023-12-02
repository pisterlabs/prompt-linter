import os 
import secret_key


os.environ["OPENAI_API_KEY"] = secret_key.open_api_key

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.llms import OpenAI

llm = OpenAI(temperature = 0)

def start_a_restaurant(Cuisine):
    name_prompt = PromptTemplate (
        input_variables= ['Cuisine'],
        template= "I want to open a {Cuisine} Restaurant. List a name you would  suggest. Give only the name"
    )

    name_chain = LLMChain(prompt = name_prompt, llm = llm, output_key = "restaurant_name")

    menu_prompt = PromptTemplate (
        input_variables= ['restaurant_name'],
        template= "Suggest a menu for {restaurant_name}. List the menu & Properly arrange the list. Give a brief description of each meal too. And an estimate price based on your Knowledge"
    )

    menu_chain = LLMChain(prompt = menu_prompt, llm = llm, output_key = "menu_list")

    augmented_chain = SequentialChain (
        chains=[name_chain, menu_chain],

        input_variables=["Cuisine"],
        output_variables=["restaurant_name", "menu_list"]
    )


    return augmented_chain({Cuisine})


    #print(augmented_chain ({"Ethiopian"}))



