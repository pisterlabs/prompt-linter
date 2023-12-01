# fmt: off
import os
import requests

from linkedin_agents import profile_lookup
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOllama, ChatOpenAI
# fmt: on


def scrape_linkedin_profile(profile_url: str):
    """Scrape information from LinkedIn profile.
    Manually scrape the information from the LinkedIn profile."""

    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {
        "Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": profile_url}, headers=header_dic
    )

    return response


if __name__ == "__main__":
    print("Hello, LangChain!")

    linkedin_data = scrape_linkedin_profile(
        profile_url="https://www.linkedin.com/in/vincent-paglioni"
    )

    print(linkedin_data.json())

    # creating summary templet
    summary_template = """
    given the {linkedin_data} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about the person
    3. few ice breakers to start conversation with the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_data"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_profile_url = profile_lookup(name="Vincent Paglioni")
    linkedin_data = scrape_linkedin_profile(profile_url=linkedin_profile_url)

    print(chain.run(linkedin_data=linkedin_data))
