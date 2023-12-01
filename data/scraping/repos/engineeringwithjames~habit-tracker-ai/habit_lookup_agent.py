from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from services.get_user_habits import get_user_habits


def lookup_habit(user_id: str) -> str:
    """Given the user's id all their habits."""

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
    )

    template = """
        given the user's id {user_id} 
        get their existing habits and if there is none just let me know 
        there is none
    """

    tools_for_agent = [
        Tool(
            name="Get the user's existing habits",
            func=get_user_habits,
            description="useful for when you need to get the user's existing habits to gain some context",
        )
    ]

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["user_id"]
    )

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )

    result = agent.run(prompt_template.format_prompt(user_id=user_id))

    return result
