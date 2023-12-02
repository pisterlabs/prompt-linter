import json
import os
from typing import Tuple

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from dotenv import dotenv_values
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from output_parsers import PersonIntel, person_intel_parser
from third_parties.linkedin import scrape_linkedin_profile

config = dotenv_values(".env")
# print(config["OPENAI_API_KEY"])
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["PROXY_CURL_API_KEY"] = config["PROXY_CURL_API_KEY"]
os.environ["SERPAPI_API_KEY"] = config["SERPAPI_API_KEY"]


# name_map
FULL_NAMES = {
    "minjoo": "Minjoo Cho",
    "bahareh": "Bahareh Sabokakin",
    "albert": "Albert Terradas",
    "joe": "Joe Hornby",
    "ken": "Ken Chan",
    "stephen": "Stephen Beckett",
    "tim": "Tim Brooke",
    "yoshi": "Yoshitsugu Kosaka",
}


def ice_break(short_name: str, mode="api") -> tuple[PersonIntel, str, str]:
    summary_template = """
    given the LinkedIn information {information} about a person from I want you to create:
    1. a short summary of a person into at most 2 sentences
    2. two interesting facts about them
    3. A topic that may interest them
    4. 2 creative Ice breakers to open a conversation with them
    \n{format_instructions}
    """

    # partial_variables : can plug in the info before the chain in running
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # run llm with a chain
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # get the full name from the short name
    full_name = FULL_NAMES[short_name]
    linkedin_data = None

    if mode == "api":
        linkedin_profile_url = linkedin_lookup_agent(name=f"{full_name} Ford")

        print(linkedin_profile_url)
        linkedin_data = scrape_linkedin_profile(
            linkedin_profile_url=linkedin_profile_url
        )

    # test example with local json file
    if mode == "local":
        f = open(f"local/{short_name}.json", "r")
        linkedin_data = json.load(f)

    result = chain.run(information=linkedin_data)
    picture = (
        linkedin_data["profile_pic_url"]
        if "profile_pic_url" in linkedin_data.keys()
        else ""
    )
    return person_intel_parser.parse(result), picture, short_name


if __name__ == "__main__":
    name = "minjoo"
    # scrape_linkedin_profile(
    #     linkedin_profile_url="https://www.linkedin.com/in/minjoo-cho-a1b2374a/",
    #     save_to_file=True,
    #     name="minjoo",
    # )
    # result = ice_break(short_name=name, mode="local")
