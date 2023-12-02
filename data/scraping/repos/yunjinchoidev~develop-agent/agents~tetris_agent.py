from tools.tools import serp_search
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """
        refer search {search} and summary 
       """

    tools_for_agent_tetris = [
        Tool(
            name="summary source code",
            func=serp_search,
            description="summary source code",
        )
    ]

    agent = initialize_agent(
        tools_for_agent_tetris,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(input_variables=["search"], template=template)

    twitter_username = agent.run(prompt_template.format_prompt(search=name))

    return twitter_username


if __name__ == "__main__":
    load_dotenv()
    print(lookup("tetris programming tutorial by pygame"))
