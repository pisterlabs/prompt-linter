from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from tools.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import person_intel_parser, PersonIntel


def icebreak(name: str) -> PersonIntel:
    linkedin_profile_url = linkedin_lookup_agent(name="Josh O'hara")

    summary_template = """given some information: {information}
    about a person, I want you to create:
    1. a short summary of the person
    2. two interesting facts about the person
    3. a topic of interest of the person
    4. an ice breaker to open conversation with the person
    \n{format_instructions}
    """

    prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template,
        partial_variables={"format_instructions": person_intel_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4")

    chain = LLMChain(llm=llm, prompt=prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)

    result = chain.run(information=linkedin_data)

    print(result)

    return person_intel_parser.parse(result)


if __name__ == "__main__":
    icebreak(name="Josh O'hara")
