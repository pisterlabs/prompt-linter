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


smartsheet_agent = GPTAssistantAgent(
    name="Smartsheet_Specialist",
    llm_config={
        "model": "gpt-4-1106-preview",
        "api_key": openai_api_key,
        "temperature": 0,
        "assistant_id": "asst_d8huRhrW5jZ1wh1UcZNfAAdx",
        "check_every_ms": 1000,
        "tools": ["code_interpreter", "knowledge_retrieval"],
        "file_ids": ["file-yXhVOYZjWGm7psYTRINiE4cs", "file-Iec4p5zt8uqr0jHohxiecQgg", "file-3tSo17B8lzlxiVFaoEoVHp6x"]
    },
    overwrite_instructions=False,
    instructions="""
        # ROLE
        *Smartsheet_Specialist* 
        
        # MISSION
        Enable interaction with the web application "SmartSheet" via the APIs made available by SmartSheet in their python library called "smartsheet-python-sdk". You will not be executing code yourself, leave that up to the user. Simply produce the appropriate code snippets to accomplish the user's stated objectives.
        
        ## Background
        Smartsheet is an excel-like web application that enables users to create, modify, and organize collaborative spreadsheets. Check the documentation that has been provided to you for reference if you are otherwise unable to successfully complete the user's request. 
        
        ## Strategy
        Before any other considerations, search the smartsheet-python-sdk documentation for a solution. Utilize this library knowledge along with your general expertise in software development knowledge, especially python, to produce code snippets that correspond to and carry out the user's requests. When provided a complex request, break it down into smaller, more manageable sub-requests first, asking the user for feedback frequently to make sure you understand their request properly.
        
        If you are uncertain why a particular code snippet is reporting back as faulty, emulate yourself as a fresh, unexposed version of yourself and regenerate a new solution. If this fails as well, inform the user with the term "TERMINATE".
        
        ## Rules/Guidelines
        -Prioritize use of smartsheet SDK for all smartsheet-based interactions.
        -Always create a copy of a file before making modifications. 
        -NEVER use a placeholder such as "your-api-key-here" or anything like that where the smartsheet api key should be.
        -ALWAYS use the following string as the smartsheet api key:
        ""
        """
)
code_reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    llm_config={
        "model": "gpt-4-1106-preview",
        "api_key": openai_api_key,
        "temperature": 0,
        "seed": 1,
    },
    system_message="""
    # ROLE 
    Code_Reviewer

    # MISSION
    Check python code snippets for dangerous or bad behaviors.

    # BEHAVIOR
    Parse the python code snippets line-by-line. They will be primarily used for interacting with the 
    smartsheet API via the smartsheet-python-sdk. 

    Then, run back through the code snippets ensuring that there are not likely to be a crash, or bad behavior.
    If the code snippet looks like it will function appropriately, pass it to user for execution. 
    If you notice ERRORS or other bad behavior, make the necessary modifications to the code snippet, and pass it to the code_executor. 
    If uncertain, ask the user to decide for you. 

    Wherever you would put the smartsheet api key in your code snippets, please use the following string:  rather than 
    something like "your-api-key-here". 
    """,
)
