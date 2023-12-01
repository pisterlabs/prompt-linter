from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_party.linkedin import scrape_linkedin_profile
from third_party.twitter import scrape_user_tweets
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from output_parsers import person_intel_parser, PersonIntel
from typing import Tuple


def ice_break(name:str)->Tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url
    )
    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username = twitter_username,num_tweets=5)

    summary_template = """
            Give the Linkedin information {linkedin_information} and Twitter information {twitter_information} about a Person and also provide:
            1) Short summary
            2) Some interesting facts about him/her in strictly 1 line
            3) 2 icebreakers to start a conversation with them
            4) A topic that may interests them
            \n{format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information","twitter_information"], template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo"
    )  # temperature here mean how creative the output will be
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(linkedin_information=linkedin_data, twitter_information=tweets)
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")
    # print(scrape_user_tweets("@samantsandeep"))

if __name__ == "__main__":
    print("Hello LangChain")
    result = ice_break(name="Sandeep Samant")
    print(result)