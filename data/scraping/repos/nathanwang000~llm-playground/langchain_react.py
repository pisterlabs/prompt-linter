import langchain
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.openai import OpenAI
from langchain.agents import load_tools, get_all_tool_names
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from lib.utils import repl
import os, click

@click.command()
@click.option('-q', 'question', prompt='Question', help='Question to ask')
def main(question):
    print(f"Tools in {langchain.agents.tools.__file__}: {get_all_tool_names()}")        
    # adapted from https://python.langchain.com/en/latest/getting_started/getting_started.html
    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    # os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
    # llm = OpenAI(temperature=0, model_name="vicuna-7b-v1.1")

    # Next, let's load some tools to use.
    tools_names = ["ddg-search", "llm-math"]
    tools = load_tools(tools_names, llm=llm)
    print(f"ask the agent anything, this is using openai's {llm.model_name} model, using {tools_names} tools")

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run(question)
    
if __name__ == "__main__":
    main()



