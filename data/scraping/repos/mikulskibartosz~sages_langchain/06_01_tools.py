# Jak użyć Langchain do interakcji z innym oprogramowaniem (Langchain agent + Langchain tools),

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import Tool


if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    wikipedia = WikipediaAPIWrapper()

    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    tools = [
        Tool.from_function(
            func=wikipedia.run,
            name = "Wikipedia",
            description="Useful when you need to find a definition of something"
        ),
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    print(agent('What is the length of the Nile?'))