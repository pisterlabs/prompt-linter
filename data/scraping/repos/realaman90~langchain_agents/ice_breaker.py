from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
import os
from dotenv import load_dotenv
from typing import Tuple

from agents.linkedin_lookup import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin

from third_parties.twitter import scrape_user_tweets
from outputparser import (
    PersonalIntel,
    person_intel_parser,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

information = """
    name: John Doe
    age: 32
    occupation: Software Engineer
    location: San Francisco
    hobbies: playing guitar, reading, writing
    """


def ice_break(name: str) -> Tuple[PersonalIntel, str]:
    if __name__ == "__main__":
        # print helloworld
        print("Hello World!")
        # print(os.environ.get('OPENAI_API_KEY'))
        linkedin_profile = linkedin_lookup_agent(name=name)

        summary_template = """
            given the information {information} about a person form I want you to create:
            1. a short summary
            2. two interesting facts about the person
            3. two topics of interest of the person (generate some random topic)
            4. two hobbies of the person (generate some random  hobby)
                \n{format_instructions}

    """

        summary_prompt_template = PromptTemplate(
            template=summary_template,
            input_variables=["information"],
            partial_variables={
                "format_instructions": person_intel_parser.get_format_instructions()
            },
        )

        llm = ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key
        )

        chain = LLMChain(llm=llm, prompt=summary_prompt_template)

        # linkedin Data
        linkedinData = scrape_linkedin(url=linkedin_profile)
        result = chain.run(information=linkedinData), linkedinData.get(
            "profile_pic_url"
        )
        print(result)

        return result


if __name__ == "__main__":
    pass
