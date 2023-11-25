import os
from pprint import pprint
from re import template
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedinDummie import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

from third_parties.twitter import scrape_user_tweets
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent

from output_parsers import person_intel_parser, PersonIntel


def ice_break(name: str) -> tuple[PersonIntel, str]:
    # linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_profile_url = "https://www.linkedin.com/jhpiedrahitao"
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    # twitter_username = twitter_lookup_agent(name=name)
    twitter_username = "jhpiedrahitao"
    tweets = scrape_user_tweets(username=twitter_username, num_tweets=5)
    summary_template = """
        given the linkedin information {linkedin_information} and twitter {twitter_information} about a person, I want you to create:
        1- a short summary
        2- two intereting facts about them
        3- A topic that may interest them
        4. 2 creative Ice brakers  to open a conversation with them
        \n{format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(linkedin_information=linkedin_data, twitter_information=tweets)
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    print("hello langchain")
    name = "Jorge Piedrahita Ortiz"
    pprint(ice_break(name))
