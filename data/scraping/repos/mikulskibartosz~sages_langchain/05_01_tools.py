# Jak użyć Langchain do interakcji z innym oprogramowaniem (Langchain agent + Langchain tools),
# https://github.com/hwchase17/langchain/issues/6083 Error in on_chain_start callback: 'name'

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    print(agent('Calculate 2+2'))