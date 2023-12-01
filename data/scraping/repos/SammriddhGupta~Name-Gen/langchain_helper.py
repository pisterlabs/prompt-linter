from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_team_name(company_name, theme_name, problem_name, idea_name):
    llm = OpenAI(temperature=1)
    
    prompt_template_name = PromptTemplate(
        input_variables = ['company_name', 'theme_name', 'problem_name', 'idea_name'], 
        template = "I'm taking part in the {company_name} hackathon, the theme is {theme_name}, the problem statement involves {problem_name} and our idea is {idea_name}. Now suggest 20 awesome team names for our group!"
    )
    
    name_chain = LLMChain(llm = llm, prompt = prompt_template_name )
    
    response = name_chain({'company_name' : company_name, 'theme_name' : theme_name, 'problem_name' : problem_name, 'idea_name' : idea_name })
    
    return response['text']

# this is for debugging only, is not considered by the webapp when you run it
if __name__ == "__main__":
    print(generate_team_name("Dolby IO", "Social Good", "using Dolby IO's APIs", "dunno yet"))