from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

def generate_pet_name( animal_type, pet_color ):

  prompt_template_name = PromptTemplate(
    input_variables = [ 'animal_type', 'pet_color' ],
    template = 'I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest me five cool names for my pet.'
  )

  name_chain = LLMChain( 
    llm = OpenAI( temperature=0.5 ),
    prompt = prompt_template_name, 
    output_key = 'pet_name'
  )

  return name_chain( 
    { 'animal_type': animal_type, 
      'pet_color': pet_color 
    } 
  )

def langchain_agent():
  
  tools = load_tools( 
    [ 'wikipedia', 'llm-math' ], 
    llm = OpenAI( temperature=0.5 ) 
  )

  agent = initialize_agent( 
    llm = OpenAI( temperature=0.5 ),
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools = tools,
    verbose = True,
  )

  result = agent.run(
    "What is average age of a dog? Multiply it by 7."
  )

  print( result )

if __name__ == '__main__':
  print( langchain_agent() )
  # print( generate_pet_name( 'cat', 'ginger' ) )