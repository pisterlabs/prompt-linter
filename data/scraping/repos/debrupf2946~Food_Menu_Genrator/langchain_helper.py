from langchain.chains import SequentialChain
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI as oa
from secret_key import api_key
import openai



api_key="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"
llm=oa(temperature=0.6,openai_api_key=api_key)

def generate(cuisine):
    temp_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want open an restraunt for {cuisine} food. Suggest me a fancy name"
    )
    name_chain2 = LLMChain(llm=llm, prompt=temp_name, output_key="restruant_name")
    temp_food = PromptTemplate(
        input_variables=["restruant_name"],
        template="return some menu items for {restruant_name}.Return a comma separated list"
    )
    food_chain2 = LLMChain(llm=llm, prompt=temp_food, output_key="menu_items")
    chain2 = SequentialChain(
        chains=[name_chain2, food_chain2],
        input_variables=["cuisine"],
        output_variables=["restruant_name", "menu_items"]

    )
    return chain2({"cuisine":cuisine})