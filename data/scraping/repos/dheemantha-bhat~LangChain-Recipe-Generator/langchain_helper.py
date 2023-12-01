from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os
os.environ['OPENAI_API_KEY'] = "Insert your OpenAI API Key"

llm = OpenAI(temperature=0.7)

def generate_dish_name_and_ingredients(diet):
    prompt_template_name = PromptTemplate(
        input_variables=['diet'],
        template="Suggest a {diet} dish. Just the name."
    )

    dish_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="dish")

    # Chain 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=['dish'],
        template="In comma separated string give me main ingredients for {dish}"
    )

    ingredient_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="ingredients")

    chain = SequentialChain(
        chains=[dish_chain, ingredient_chain],
        input_variables=['diet'],
        output_variables=['dish', "ingredients"]
    )

    response = chain({'diet': diet})

    return response

if __name__ == "__main__":
    print(generate_dish_name_and_ingredients("Vegan"))
