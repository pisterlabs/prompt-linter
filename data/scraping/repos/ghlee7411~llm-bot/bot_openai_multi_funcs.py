from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from tools import mermaid, search
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)
del load_dotenv

# for debugging
import langchain
langchain.debug = False


def main():
    llm = ChatOpenAI(temperature=0)
    tools = [
        Tool(
            name="mermaid_diagram_generator",
            func=mermaid,
            description="You can depict a situation using a Mermaid diagram, illustrating the scenario in visual form.",
        ),
        Tool(
            name="search_engine",
            func=search,
            description="You can ask any question and get the answer.",
        ),
    ]

    mrkl = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True,
    )

    output = mrkl.run("How to cook a pizza? Describe the process and the ingredients. Additionally, you can add a diagram.")

    print('[Output]')
    print(output)


if __name__ == "__main__":
    main()