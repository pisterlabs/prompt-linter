from enum import Enum
from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from output_parsers import person_info_parser, PersonInfo
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets


class Strategy(Enum):
    TWITTER = 1
    LINKEDIN = 2


def get_prompt_template(prefix: str) -> PromptTemplate:
    default_prompt_template = """
        1. a short summary
        2. two interesting facts about them
        3. a topic that may interest them
        4. 2 creative ice breakers to open a conversation with them
        \n{format_instructions}
    """
    return PromptTemplate(
        input_variables=["information"],
        template=prefix + default_prompt_template,
        partial_variables={
            "format_instructions": person_info_parser.get_format_instructions()
        }
    )


def linkedin(name: str) -> Tuple[str, PersonInfo]:
    summary_prompt_template = get_prompt_template("given the LinkedIn information {information} about a person I want "
                                                  "you to create:")

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_profile_url = linkedin_lookup_agent(name=name)
    print(linkedin_profile_url)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url,
        use_cache=True
    )
    return linkedin_data.get("profile_pic_url"), person_info_parser.parse(chain.run(information=linkedin_data))


def twitter(name: str) -> Tuple[str, PersonInfo]:
    summary_prompt_template = get_prompt_template("given the Twitter information {information} about a person I want "
                                                  "you to create:")

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    twitter_username = twitter_lookup_agent(name=name)
    print(twitter_username)
    twitter_data = scrape_user_tweets(username=twitter_username, num_tweets=10)
    return twitter_username, person_info_parser.parse(chain.run(information=twitter_data))


def ice_break(name: str, strategy: Strategy) -> PersonInfo | None:
    result: PersonInfo
    if strategy == Strategy.LINKEDIN:
        _, result = linkedin(name)
    elif strategy == Strategy.TWITTER:
        _, result = twitter(name)
    else:
        return None
    return result


if __name__ == "__main__":
    print(ice_break(name="Bill Gates", strategy=Strategy.LINKEDIN))
