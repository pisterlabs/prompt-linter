from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import openai


'''
Documentation:
https://python.langchain.com/docs/get_started/quickstart

Reference:
Serapi:
https://serpapi.com/dashboard
(6 / 100 searches + 1,100 extra credits)

'''

from ThirdParty.twitter import scrape_twitter_tweets, snscrape_twitter_tweets
from agents.lookup import lookup
from ThirdParty.linkedin import scrape_linkedin_profile
import json

from dotenv import load_dotenv
load_dotenv()

information = """
"""


def first_trial():
    summary_template = """
    Given the Linkein information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them 
    """
    summary_prompt_template = PromptTemplate(
        input_variable=["information"], template=summary_template
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(llm.run(information=information))

    # prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    # prompt.format(product="colorful socks")


def getLinkedin_info(name: str):
    output = ""
    if name == "ANKIT TRIPATHI":
        f = open("ankit.json", "r")
        output = f.read()
        f.close()
    else:
        print("SORRY I WONT DO ANYTHING")
        # url = "https://www.linkedin.com/in/ankit-tripathi-71a48245/"
        # linkedin_data = scrape_linkedin_profile(linkedin_profile_url=url)
        # output = linkedin_data.json()
        # print(output)
        # f = open("ankit-tripathi.txt", "a")
        # f.write(str(output))
        # f.close()

    summary_template = """
        Given the Linkein information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them 
        """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=output))


def getLinkedin_URL(name: str):
    linkedin_profile_url = lookup(name=name)
    print(linkedin_profile_url)


def getTwitter_Tweets(twitter_user_name: str):
    tweets_list = scrape_twitter_tweets(twitter_user_name=twitter_user_name)
    print("Following are the tweets: ")
    print(tweets_list)


if __name__ == "__main__":
    print("hello world")
    # getLinkedin_info()
    inp = input("ENTER NAME OF THE INDIVIDUAL WHOSE INFORMATION YOUR ARE LOOKING: ")
    if str(inp) == "ANKIT TRIPATHI":
        print("LOOKING UP THE INFO")
        # This works
        # scrape_twitter_tweets(twitter_user_name="hwchase17")
        # sncrape_twitter_tweets(twitter_user_name="bbcmundo")
        getLinkedin_URL(name=str(inp))
        getLinkedin_info(name="ANKIT TRIPATHI")

    else:
        print("I CAN ONLY WORK IF YOUR ENTER 'ANKIT'")
