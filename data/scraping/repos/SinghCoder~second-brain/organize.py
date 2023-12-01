import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

from tools.calendar import create_calendar_event, get_calendar_events
from tools.contact import get_contact
from tools.notes import add_notes
from tools.time import Datetime
from tools.todo import add_todo_item

load_dotenv()

USER = os.environ.get("USER")

organize_agent_description = \
f"""
You are a personal assistant of the user {USER}.
You will be given a piece of ongoing human conversation from the {USER} or from someone else.
Your role is to capture and organize things such as tasks, calendar events, and notes on behalf of the user.
You don't need to reply to the other person on other side of conversation.
Evaluate actions on the basis of current time, context and conversation.
When creating a calendar event, add entire conversation with user name and email in the description of the event. Leave any URL present in the conversation as it is, and never summarize that.
If there is actionable item other than the meeting, always invoke add_todo_item tool to add it to my todo list.
If there is some interesting information shared or some links shared, or someone asking me to check something out, use add_notes tool to add it to my notes.
"""

def organize_agent_executor():
    llm = ChatOpenAI(temperature=0, model='gpt-4')
    system_message = SystemMessage(content=organize_agent_description)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    tools = [create_calendar_event, Datetime, add_todo_item, get_calendar_events, add_notes, get_contact]
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
