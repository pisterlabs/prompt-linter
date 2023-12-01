from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from output_parser import person_data_parser 


def ice_text(name: str) -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    linkedin_profile_url = linkedin_lookup_agent(name="Pranav Ajay Vit Bhopal")

    summary_template = """
    given the information{information} of a person from I want you to create:
    1. a short summary
    2. two interesting facts about them 
     \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template,
        partial_variables={"format_instructions":person_data_parser.getf_format_instructions() }
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linked_in_url=linkedin_profile_url)

    result = chain.run(information=linkedin_data)

    return person_data_parser.parse(result)


if __name__ == "__main__":
    result = ice_text()
