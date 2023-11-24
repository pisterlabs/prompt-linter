# Importing the PromptTemplate class from the langchain module
from langchain import PromptTemplate
# Importing the ChatOpenAI class from the langchain.chat_models module
from langchain.chat_models import ChatOpenAI
# Importing the LLMChain class from the langchain.chains module
from langchain.chains import LLMChain

import os
import requests

from third_parties.linkedin import scrape_linkedin_profile

information = """
"""

if __name__ == "__main__":
    print("Hello Langchain")

    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile()

    print(chain.run(information=linkedin_data))
