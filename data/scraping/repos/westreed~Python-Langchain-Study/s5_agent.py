from key import APIKEY
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI


if __name__ == "__main__":
    KEY = APIKEY()
    # The language model we're going to use to control the agent.
    llm = OpenAI(openai_api_key=KEY.openai_api_key, temperature=0)

    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=KEY.serpapi_api_key)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Let's test it out!
    res = agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
    print(res)