from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os
import secrete  

llm = OpenAI(openai_api_key=secrete.OPEN_API_KEY ,temperature=0.7)

def get_car_models_and_colors(company_name, vehicle_type):
    # Define the prompt template for Car Models
    prompt_template_models = PromptTemplate(
        input_variables=['company_name', 'vehicle_type'],
        template="""Suggest the top 10 {vehicle_type} models for {company_name} with different colors."""
    )
    models_chain = LLMChain(llm=llm, prompt=prompt_template_models, output_key="car_models")

    # Define the sequential chain
    chain = SequentialChain(
        chains=[models_chain],
        input_variables=["company_name", "vehicle_type"],
        output_variables=["car_models"],
    )

    # Generate the response
    response = chain({"company_name": company_name, "vehicle_type": vehicle_type})
    return response

# if __name__ == "__main__":
#     company_name = input("Enter the car company name: ")
#     vehicle_type = input("Enter the type of vehicles (two wheeler or four wheeler): ")
#     result = get_car_models_and_colors(company_name, vehicle_type)