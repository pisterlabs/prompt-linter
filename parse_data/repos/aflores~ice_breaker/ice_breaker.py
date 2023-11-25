from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

if __name__ == '__main__':
    print("Hello LangChain!")


# information="""
# Christopher Chiyan Tin (born May 21, 1976) is an American composer of art music, often composed for film and video game soundtracks. His work is primarily orchestral and choral, often with a world music influence. He won two Grammy Awards for his classical crossover album Calling All Dawns.

# Tin is perhaps best known for his choral piece Baba Yetu from the video game Civilization IV, which in 2011 became the first piece of video game music to win a Grammy Award.[1] His Grammy win was considered a significant milestone for the critical acceptance of music from video games as a legitimate art form, and following his win the Recording Academy retitled their visual media categories to become more inclusive of video game soundtracks,[2] before eventually creating a dedicated Grammy award for 'Best Score Soundtrack for Video Games and Other Interactive Media'
# """


# linkedin_profile_url='https://www.linkedin.com/in/armando-flores-0570861')
linkedin_profile_url=linkedin_lookup_agent(
    name="Armando Flores Miami Florida")
 
summary_template = """
    given the LinkedIn information {information} about a person from I want you to create
    1. a short summary
    2. two interesting facts about about this person
    """

summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")

chain = LLMChain(llm=llm, prompt=summary_prompt_template)


linkedin_data = scrape_linkedin_profile(
    linkedin_profile_url = linkedin_profile_url)

print(chain.run(information=linkedin_data))
