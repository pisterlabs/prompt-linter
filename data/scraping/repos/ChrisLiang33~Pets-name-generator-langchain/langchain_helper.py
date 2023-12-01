from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet and I want a cool name for it, is it {pet_color}, suggest me 5 cool name for my pet"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='pet_name')

    response = name_chain({'animal_type': animal_type, 'pet_color':pet_color})
    return response
if __name__ == '__main__':
    print(generate_pet_name('cow', 'black'))