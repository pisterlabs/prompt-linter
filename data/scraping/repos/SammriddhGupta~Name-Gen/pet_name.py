from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(pet_colour, animal_type):
    llm = OpenAI(temperature=0.7)
    
    prompt_template_name = PromptTemplate(
        input_variables = ['pet_colour', 'animal_type'], 
        template = "I have a {pet_colour} pet {animal_type} and want a funky name for it, gimme 10 funky names!"
    )
    
    name_chain = LLMChain(llm = llm, prompt = prompt_template_name )
    
    response = name_chain({'pet_colour' : pet_colour, 'animal_type' : animal_type, })
    
    return response

if __name__ == "__main__":
    print(generate_pet_name("white", "kangaroo"))