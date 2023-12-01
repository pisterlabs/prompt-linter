from keys import mykey

import os

from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = mykey

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
llm = OpenAI(temperature=0.6)

def gen_rest_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
    input_variables= ['cuisine'],
    template="I want to open a restuarant for {cuisine} food. Suggest a fancy name for it"
)
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")


    prompt_template_items = PromptTemplate(
        input_variables= ['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma seperated list"
    )


    food_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key= "menu_items")

    chain = SequentialChain(
        chains= [name_chain, food_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', "menu_items"]
    )

    response = chain({"cuisine": cuisine})
    return response
