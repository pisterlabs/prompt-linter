from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile


if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
       Given an {information} about a person, please provide the following
       1) A brief summary of the person
       2) Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # llm = ChatOpenAI(temperature=0, model="gpt-4")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url="https://gist.githubusercontent.com/emarco177"
    #     "/0d6a3f93dd06634d95e46a2782ed7490/raw"
    #     "/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden"
    #     "-marco.json"
    # )

    # Get the URL using the lookup agent in agents
    linkedin_profile_url = linkedin_lookup_agent(name="Ramesh Venkatraman HCL")

    # Scrape the URL using the scrape_linkedin_profile function from linkedin.py under third_parties
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    print(chain.run(information=linkedin_data))
