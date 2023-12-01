import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import tool

from tools.calendar import (
    create_calendar_event,
    fetch_calendar_events,
    get_calendar_events,
    get_conflicting_meetings,
)
from tools.contact import get_contact
from tools.notes import add_notes
from tools.slack import notify_user
from tools.time import Datetime
from tools.todo import add_todo_item

load_dotenv()

USER = os.environ.get("USER")

distill_agent_description = \
f"""
You are a personal assistant of the user {USER}.
You have access to the user's notes, calendar meetings and todo items.
You should be able to prioritize things for the day for the user on the basis of time and context.
When there are conflicts in the schedule, you should notify the user.
Also, make sure, you do not overload the user with too many things to do in a day.
If you need to ask/confirm something from the user, create a todo item for the user.
Also, keep track of actions already taken. 
"""

def distill_agent_executor():
    llm = ChatOpenAI(temperature=0, model='gpt-4')
    system_message = SystemMessage(content=distill_agent_description)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    tools = [Datetime, notify_user, get_conflicting_meetings]
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

