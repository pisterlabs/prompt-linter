from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.tools import tool, BaseTool
from langchain.tools.base import ToolException
from pydantic import BaseModel, Field
import requests
import json

from robocorp.tasks import task
from robocorp import vault

from RPA.Assistant.types import WindowLocation, Size
import RPA.Assistant

ws_secret = vault.get_secret("LangchainWS")
openai_secret = vault.get_secret("OpenAI")

llm = OpenAI(temperature=0, openai_api_key=openai_secret["key"])

assistant = RPA.Assistant.Assistant()
gpt_conversation_display = []
gpt_conversation_internal = []

memory = ConversationBufferMemory(memory_key="chat_history")

def _handle_error(error: ToolException) -> str:
    message = (
        "The following errors occurred during tool execution:"
        + str(error) + "\n"
        + "Please try another tool."
    )

    gpt_conversation_display.append((None, message))

    return message

class ToolInputSchema(BaseModel):
    name: str = Field(description="should be a lead name")

def add_lead(params):
    """Starts the execution of the bot that fills in a new contact form in CRM with one lead company/person details."""
    print(f"Received params: {params}")
    return {"response": "Lead added successfully!"}

def list_available_processes(params):
    """Lists all available processes."""
    print(f"Received params: {params}")
    response = requests.get(
        f"https://api.eu1.robocorp.com/process-v1/workspaces/{ws_secret['ws-id']}/processes",
        headers={
            "Authorization": f"RC-WSKEY {ws_secret['key']}"
        }
    )
    data = response.json()
    print('👀 this is data', data["data"])
    return "\n".join(map(lambda p: f"Process name: {p['name']}, Id: {p['id']}", data["data"]))

def start_process(process_id: str):
    """Start process"""
    print('👀 this is process', process_id)
    response = requests.post(
        f"https://api.eu1.robocorp.com/process-v1/workspaces/{ws_secret['ws-id']}/processes/{process_id}/run-request",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"RC-WSKEY {ws_secret['key']}"
        },
        data=json.dumps({
            "type": "default"
        })
    )
    data = response.json()

    if "error" in data:
        raise ToolException(data["error"]["message"])

    print('👀 this is data', data)

    return f"Process started, run id: {data['id']}"

add_lead_tool = Tool(
    name="Add Lead Tool",
    func=add_lead,
    description="Adds a new lead to CRM, mandatory input parameters are name and email",
    handle_tool_error=_handle_error,
)

list_available_processes_tool = Tool(
    name="List workspace processes",
    func=list_available_processes,
    description="Lists available workspace processes",
    handle_tool_error=_handle_error,
)

start_process_tool = Tool(
    name="Start a process run",
    func=start_process,
    description="Starts a process run",
    handle_tool_error=_handle_error,
)

tools = [
    add_lead_tool,
    list_available_processes_tool,
    start_process_tool,
]

# replace the default prompt template by overriding the agent's llm_chain.prompt.template
# print(agent.agent.llm_chain.prompt.template)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def show_spinner():
    assistant.clear_dialog()
    assistant.add_loading_spinner(name="spinner", width=60, height=60, stroke_width=8)
    assistant.refresh_dialog()


def ask_gpt(form_data: dict):
    text = agent_chain.run(input=form_data["input"])
    gpt_conversation_display.append((form_data["input"], text))

    display_conversation()
    assistant.refresh_dialog()

def display_conversation():
    assistant.clear_dialog()
    assistant.add_heading("Conversation")

    for reply in gpt_conversation_display:
        if reply[0] is not None:
            assistant.add_text("You:", size=Size.Small)
            assistant.open_container(background_color="#C091EF", margin=2)
            assistant.add_text(reply[0])
            assistant.close_container()

        if reply[1] is not None:
            assistant.add_text("GPT:", size=Size.Small)
            assistant.open_container(background_color="#A5AACD", margin=2)
            assistant.add_text(reply[1])
            assistant.close_container()

    display_buttons()

def display_buttons():
    assistant.add_text_input("input", placeholder="Send a message", minimum_rows=3)
    assistant.add_next_ui_button("Send", ask_gpt)
    assistant.add_submit_buttons("Close", default="Close")

@task
def run_chat():

    display_conversation()

    assistant.run_dialog(
        timeout=1800, title="AI Chat", on_top=True, location=WindowLocation.Center
    )