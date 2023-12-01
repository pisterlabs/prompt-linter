from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

def generate_pet_name(animal_type, animal_color):
    llm= OpenAI(temperature=0.6)
    #name=llm("I have a dog and I want a name for it. please suggest me cool name for it")
    #return name
    prompt_name = PromptTemplate(
        input_variables = ['animal_type','animal_color'],
        template = "I have a {animal_type} and its  color {animal_color}. So, I want a name for it. please suggest me cool name for it",
    )
 
    name_chain = LLMChain(llm=llm, prompt=prompt_name, output_key="pet_name")

    response = name_chain({'animal_type': animal_type, 'animal_color': animal_color})
    return response

if __name__ == "__main__":
    print(generate_pet_name("Dog","Black"))