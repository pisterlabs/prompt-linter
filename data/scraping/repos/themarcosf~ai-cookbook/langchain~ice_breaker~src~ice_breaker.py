from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import linkedin_lookup_agent
from output_parsers.output_parsers import PersonIntel, person_intel_parser


def ice_break(name: str, reference: object) -> PersonIntel:
    linkedin_profile_url = linkedin_lookup_agent(
        name=name,
        reference=reference,
    )

    summary_template = """
        Given the LinkedIn information {information} about a person, I want you to create:
        1. a short summary
        2. two interesting facts about the person
        3. a topic that might interest the person
        4. two creative ice breakers to start a conversation with the person
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
        temperature=0.0,
        model="gpt-4",
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    data = scrape_linkedin_profile(linkedin_profile_url)

    response = chain.run(information=data)

    return response
