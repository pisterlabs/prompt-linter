import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_yelp_business_id

def lookup(name: str, address1: str, city: str, state: str, country: str = 'US') -> str:
    llm = ChatOpenAI(temperature=0, model_name=os.environ.get('MODEL_NAME'))
    template = """given the name {name} and address1 {address1} and city {city} and state {state} and country {country} I want you to determine if it is still in business.
       Your answer should contain only a string"""

    tools_for_agent = [
        Tool(
            name="Crawl Yelp 4 business id",
            func=get_yelp_business_id,
            description="useful for when you need to know if a business is still in business",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=['name', 'address1', 'city', 'state', 'country']
    )

    bstatus = agent.run(prompt_template.format_prompt(name=name, address1=address1, city=city, state=state,
                                                               country=country))

    return bstatus
