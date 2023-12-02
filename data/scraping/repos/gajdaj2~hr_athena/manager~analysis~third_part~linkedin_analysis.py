from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

from manager.analysis.third_part.linkedin import scrape_linkedin_profile

#read env file
from dotenv import load_dotenv

load_dotenv()
def summary_template():
    return """
    given the linkedin information {information} about person  from I want you to create:
        1. a short summary of the person
        2. 5 Technology what he know
        3. Three interview questions and answers you can ask him base on his technical skills
    """

def linkedin_analysis(link):
    linkedin_profile = scrape_linkedin_profile(link)
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template())
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    profile = chain.run(information=linkedin_profile)
    return profile

if __name__ == '__main__':
    print(linkedin_analysis("https://www.linkedin.com/in/alexander-gajdaj-2b0b0b1b3/"))