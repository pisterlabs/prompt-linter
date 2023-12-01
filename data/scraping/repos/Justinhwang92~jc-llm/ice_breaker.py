from typing import Tuple
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedIn import scrape_linkedin_profile
from output_parsers import (person_intel_parsers, PersonIntel)


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name)

    summary_template = """
        Given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about the person
        3. A topic that may interest the person
        4. 2 creative Ice breakers to open a conversation with the person 
        \n{format_instruction}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instruction": person_intel_parsers.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url)
    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url=linkedin_profile_url, debug=True)

    result = chain.run(information=linkedin_data)

    return person_intel_parsers.parse(result), linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    result = ice_break(name="Soonkwon Hwang")
    print(result)
