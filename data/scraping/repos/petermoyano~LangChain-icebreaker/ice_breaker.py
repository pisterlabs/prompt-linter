from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# import os

# Custom packages (agent and tools for the agent)
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile


if __name__ == "__main__":
    print("Ice Breaker App started")
    # print(os.environ["OPENAI_API_KEY"])

    # Agent execution: name: str -> str
    # This agent will return a linkedin profile url starting with a name
    linkedin_profile_url = linkedin_lookup_agent(
        name="Luis Gregorio Moyano")

    summary_template = """
        Given the LinkedIn information {information} about a person, I want you to create:
        1. a short joke about the person
        1. a short summary of the person
        2. two interesting facts about the person
    """

    # Langchain automatically passes the information variable to the prompt
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm inizialization
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Chain execution
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # This function uses the Proxycurl API to scrape information from a LinkedIn profile
    # and returns a cleaned dict of the scraped data.
    # scrape_linkedin_profile: linkedin_profile_url: str -> dict
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
