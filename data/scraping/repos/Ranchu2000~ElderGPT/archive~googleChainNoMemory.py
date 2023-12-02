from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic.v1 import BaseModel, Field
from langchain import PromptTemplate
from langchain.callbacks import HumanApprovalCallbackHandler

from GoogleCalendar.googleCalendar import *

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API= os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY= os.getenv('SERPAPI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class DeleteCalendarEventInput(BaseModel):
    event: str = Field(description="Calendar event id of event to delete")

from langchain.tools import StructuredTool #tools from functions
createEventTool= StructuredTool.from_function(create_calendar_event)
listEventTool= StructuredTool.from_function(list_calendar_events)
deleteEventTool= StructuredTool.from_function(deleteEvent)
currentDateTimeTool= StructuredTool.from_function(currentDateTime)

createEventTool.callbacks= [HumanApprovalCallbackHandler()] #human approval requirememt

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
        args_schema=SearchInput
    ),
]
tools.append(createEventTool)
tools.append(deleteEventTool)
tools.append(listEventTool)
tools.append(currentDateTimeTool)

from langchain.tools.render import format_tool_to_openai_function #to observe what is passed to agent
# functions=[format_tool_to_openai_function(t) for t in tools] #tools include the functions to run
# print(tools)
# print(functions)#openai_functions are the function description but they are not very detailed

# Using OpenAIFunctionsAgent- AgentType.OPENAI_FUNCTIONS
prompt="""
    You are a helpful assistant that manages Tom's calendar events. 
    Do not assume the current date or any of the function's arguments.
    Today's date is 25th October 2023.
"""
agent_executor = initialize_agent(tools, 
                                  llm=llm, 
                                  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
                                  verbose=False,
                                  agent_kwargs={
                                      'prefix': prompt
                                  })
response= agent_executor.invoke({"input": "I would like to rockclimb this coming Tuesday from 1pm to 3pm"})