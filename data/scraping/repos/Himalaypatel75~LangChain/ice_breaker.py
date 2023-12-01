from typing import Tuple
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import random
import os
from dotenv import load_dotenv
import requests
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsors import PersonIntel, person_intel_parser
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scraper_user_tweets
from third_parties.twitter_with_stubs import scrape_user_tweets

load_dotenv()  # loading environment variable from current dictionary


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY"
    )  # store environment variable in .env file

    # ----------Getting LinkedIn URL with Agent---------- #
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    twitter_username = twitter_lookup_agent(name=name)
    tweets_data = scrape_user_tweets(username=twitter_username, num_tweets=100)

    # ----------With URL get Profile Details---------- #
    summary_template = """
    given the linkedin information {linkedin_information} and twitter {twitter_information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them
    3. A topic that may interest them
    4. 2 creative Ice breakers to open a conversation with them
    \n {format_instructions}"""

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo"
    )  # temperature will decide how much llm model can be creative. 0 is less creative. 1 will be creative only 2 values can be

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    result = chain.run(
        linkedin_information=linkedin_data, twitter_information=tweets_data
    )
    # print(result)
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    name = "Eden Marco"
    result = ice_break(name)
    print(result)
