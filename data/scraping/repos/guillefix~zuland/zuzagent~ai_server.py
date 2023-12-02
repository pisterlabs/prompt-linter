from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
# from langchain.memory import ConversationBufferMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore

import http.server
import socketserver
from threading import Event

import re
import os

import aiohttp
import asyncio
from aiohttp import web

import json

# remove me

#%%

''' CONNECTION HANDLER '''

## Server connections events and handler

class GlobalRequestHandler:
    def __init__(self):
        self.event1 = Event()
        self.event2 = Event()
        self.is_first_action = True
        self.observation = None

    def set_observation(self, data):
        self.observation = data
        self.event1.set()

    def get_observation(self):
        self.event1.wait()
        self.event1.clear()
        return self.observation

    def set_action(self, data):
        if self.is_first_action:
            self.is_first_action = False
        else:
            self.action = data
            self.event2.set()

    def get_action(self):
        self.event2.wait()
        self.event2.clear()
        return self.action

global_request_handler = GlobalRequestHandler()

obs_template = """[
Location: {location}
Activity: {activity}
Nearby players: {nearby_players}
Objects you have in the inventory: {inventory}
You have just been told: {message}
Funds you have: {funds}
]
"""

async def handle_post(request):
    global global_request_handler

    post_data = await request.text()
    obs_json = json.loads(post_data)

    observation = obs_template.format(**obs_json)
    global_request_handler.set_observation(observation)

    action = global_request_handler.get_action()

    return web.Response(text=json.dumps(action))

app = web.Application()
app.router.add_post('/', handle_post)

#%%
''' ACTIONS '''

def send_action_and_wait_for_obs(action_name, action_input):
    actionArgs = action_input.split("|")
    action = {
        "action": action_name,
        "actionArg1": actionArgs[0] if len(actionArgs) > 0 else "",
        "actionArg2": actionArgs[1] if len(actionArgs) > 1 else  "",
    }
    global_request_handler.set_action(action)
    observation = global_request_handler.get_observation()
    return observation


# Define which tools the agent can use to answer user queries
movement_actions = [
    Tool(
        name=f"move-{direction}",
        func=(lambda direction=direction: (lambda inp: send_action_and_wait_for_obs(f"move-{direction}", "")))(),
        description=f"an action that makes you move {direction}"
    )
    for direction in ["up", "down", "left", "right"]
]

# use_action = Tool(
#     name=f"use",
#     func = lambda inp: f"You used {inp}",
#     description="an action that lets you use an object"
# )

travel_action = Tool(
    name=f"travelTo",
    func = lambda inp: send_action_and_wait_for_obs("travelTo", inp),
    description="an action that teleports you to the location. The Action Input should be the location to travel to."
)

talk_action = Tool(
    name=f"talkTo",
    func = lambda inp: send_action_and_wait_for_obs("talkTo", inp),
    description="an action to talk with another player. The Action Input should have the form `NPCName|Message`"
)

give_action = Tool(
    name=f"GiveObject",
    func = lambda inp: send_action_and_wait_for_obs("GiveObject", inp),
    description="an action to give an object to another player. The Action Input should have the form `NPCName|ObjectName`"
)

send_action = Tool(
    name=f"SendMoney",
    func = lambda inp: send_action_and_wait_for_obs("SendMoney", inp),
    description="an action to send money to another player. The Action Input should have the form `NPCName|MoneyAmount`"
)

create_art_action = Tool(
    name=f"CreateArt",
    func = lambda inp: send_action_and_wait_for_obs("CreateArt", inp),
    description="an action to create a new piece of art"
)

# ALL_ACTIONS_SEARCH = movement_actions + [travel_action, talk_action, give_action, send_action, create_art_action]
ALL_ACTIONS_SEARCH = movement_actions + [travel_action, talk_action]

# Tool Retriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_ACTIONS_SEARCH)]

actions_vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

actions_retriever = actions_vector_store.as_retriever()

def get_action(query):
    docs = actions_retriever.get_relevant_documents(query)
    actions = [ALL_ACTIONS_SEARCH[d.metadata["index"]] for d in docs]
    return actions[0]

def dummy_action_func(input):
    regex = r"Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    match = re.search(regex, input, re.DOTALL)
    if not match:
        action_input = ""
    else:
        action_input = match.group(1)
    if input != "":
        action = get_action(input)
        return action(action_input)
    else:
        return ""

dummy_action = Tool(
    name=f"Dummy action",
    func=lambda inp: dummy_action_func(inp),
    description=f"the agent didn't produce a good action"
)

ALL_ACTIONS = ALL_ACTIONS_SEARCH + [dummy_action]
#%%

''' MEMORY '''

# MEMORY
from langchain.schema import Document

docs = [""]
vector_store = FAISS.from_texts(docs, OpenAIEmbeddings())
# vector_store.add_texts(["awawawa"])
retriever = vector_store.as_retriever()

def get_memories(query):
    docs = retriever.get_relevant_documents(query)
    return ". ".join([d.page_content for d in docs])

# get_memories("fruit")

''' AGENT '''

# Set up the base template
template = """You are a player in an RPG sim world. You are able to perform the following actions:

{actions}

Use the following format:

Background: your background history, according to which to interact in the game
Memories: results queried from memory relevant to the recent interactions
Thought: you should always think about what to do
Action: the action to take, should be one of [{action_names}]
Action Input: the input to the action (could be an empty string)
Observation: the result of the action. Can have multiple lines between brackets
... (this Thought/Action/Action Input/Observation/Memories repeats forever)

Begin!

Background: {background}
Memories: {memory}
{agent_scratchpad}"""


MAX_INTERMEDIATE_STEPS = 10

get_actions = lambda x: ALL_ACTIONS # dummy

from typing import Callable
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    actions_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps[-MAX_INTERMEDIATE_STEPS:]:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        if len(intermediate_steps) > 0:
            last_action, last_observation = intermediate_steps[-1]
            memory = "You did: "+last_action.log+". You observed: "+last_observation
            memory = memory.replace("\n", ". ")
            vector_store.add_texts([memory])
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # print(thoughts)
        if len(thoughts) > 0:
            recent_thoughts = ""
            for action, observation in intermediate_steps[-5:]:
                recent_thoughts += action.log
                recent_thoughts += f"\nObservation: {observation} "
            memories = get_memories(recent_thoughts)[:1]
            # print("Memories: "+memories)
        else:
            memories = ""
        kwargs["memory"] = memories
        ############## NEW ######################
        actions = self.actions_getter(kwargs["background"])
        # Create a tools variable from the list of tools provided
        kwargs["actions"] = "\n".join([f"{action.name}: {action.description}" for action in actions if action.name != "Dummy action"])
        # Create a list of tool names for the tools provided
        kwargs["action_names"] = ", ".join([action.name for action in actions if action.name != "Dummy action"])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    actions_getter=get_actions,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    # input_variables=["background", "interaction_history", "intermediate_steps"]
    input_variables=["background", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            print(f"Could not parse LLM output: `{llm_output}`. Trying tool search")
            action = "Dummy action"
            action_input = "llm_output"
        else:
            action = match.group(1).strip()
            action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

#%%

# Set up LLM, stop sequence, and the agent

llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

actions = get_actions("whats the weather?")
action_names = [action.name for action in actions]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=action_names
)

''' MAIN FUNCTIONS '''

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=actions, verbose=True, max_iterations = 100)

async def run_server(app, host='localhost', port=8080):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f'Serving on {host}:{port}')

    return runner

async def stop_server(runner):
    await runner.cleanup()

async def run_agent_executor_async(agent_executor):
    # If agent_executor.run() is synchronous, run it in a separate thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, agent_executor.run, "You are a music shop owner")

    # If agent_executor.run() is asynchronous, you can call it directly:
    # await agent_executor.run("You are a music shop owner")

async def main():
    # Instantiate agent_executor here if needed (e.g., agent_executor = AgentExecutor())

    server_runner = await run_server(app)
    await run_agent_executor_async(agent_executor)
    await stop_server(server_runner)

if __name__ == "__main__":
    asyncio.run(main())
