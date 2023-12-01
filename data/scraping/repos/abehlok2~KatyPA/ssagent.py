import os
from openai import OpenAI
import autogen
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen.oai import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
import smartsheet

openai_api_key = "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"
print(openai_api_key)
smartsheet_api_key = ""

config_list = config_list_from_json("/home/abehl/ssagent/OAI_CONFIG_LIST.json")
user = autogen.UserProxyAgent(
    name="user",
    code_execution_config = {"use_docker": False},
    max_consecutive_auto_reply=5,
    human_input_mode="NEVER",
)


request = """
Using the sheet called "Campaign Master Lists" as a template, create a new sheet called CoreHeat-25A. 
"""

request2 = """
How many sheets do I have available to me in smartsheet?
Utilize this API key in all code snippets that you produce. Do not use a placeholder such as "your-api-key-here" or anything like that
Specifically use the following string:\n"7GJWGNGb7PaQ28tcSHYbwD1PkysZbzVHyv8dF"
"""

user.initiate_chat(smartsheet_agent, message=request)


# TODO: add a way to parse and provide the folder, file ids from smartsheet.





