# ### Initialize Agents V3
################################################################################

# Importing necessary modules and classes for LangChain Agent
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory  # Make sure to import this correctly
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

# Set up environment
# Import necessary modules and packages to set up the LLM agent.

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

from langchain.chains import LLMChain, ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.mapreduce import MapReduceChain, MapReduceDocumentsChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import BaseLLM, LlamaCpp, GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, DeepLake
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager

from langchain_experimental.autonomous_agents import BabyAGI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re



from langchain.tools import MoveFileTool
from langchain.tools import FileSearchTool
from langchain.tools import ListDirectoryTool
from langchain.tools import ReadFileTool
from langchain.tools import WikipediaQueryRun
from langchain.tools import ArxivQueryRun
from langchain.tools import CopyFileTool
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import JsonListKeysTool
from langchain.tools import WriteFileTool
from langchain.tools import ClickTool
from langchain.tools import VectorStoreQATool
from langchain.tools import DeleteFileTool
from langchain.tools import ExtractTextTool
from langchain.tools import ExtractHyperlinksTool
from langchain.tools import GetElementsTool
from langchain.tools import NavigateBackTool
import langchain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser

tools = [
    Tool(
        name="MoveFileTool",
        func=langchain.tools.file_management.move.MoveFileTool.run,
        description="Move files from one location to another"
    ),
    Tool(
        name="FileSearchTool",
        func=langchain.tools.file_management.file_search.FileSearchTool.run,
        description="Search for files in a subdirectory that match a regex pattern"
    ),
    Tool(
        name="ListDirectoryTool",
        func=langchain.tools.file_management.list_dir.ListDirectoryTool.run,
        description="List files and directories in a specified folder"
    ),
    Tool(
        name="ReadFileTool",
        func=langchain.tools.file_management.read.ReadFileTool.run,
        description="Read content from a file"
    ),
    Tool(
        name="WikipediaQueryRun",
        func=langchain.tools.wikipedia.tool.WikipediaQueryRun.run,
        description="Query information from Wikipedia"
    ),
    Tool(
        name="ArxivQueryRun",
        func=langchain.tools.arxiv.tool.ArxivQueryRun.run,
        description="Query information from Arxiv"
    ),
    Tool(
        name="CopyFileTool",
        func=langchain.tools.file_management.copy.CopyFileTool.run,
        description="Copy files from one location to another"
    ),
    Tool(
        name="DuckDuckGoSearchResults",
        func=langchain.tools.ddg_search.tool.DuckDuckGoSearchResults.run,
        description="Search for information using DuckDuckGo"
    ),
    Tool(
        name="JsonListKeysTool",
        func=langchain.tools.json.tool.JsonListKeysTool.run,
        description="List keys in a JSON object"
    ),
    Tool(
        name="WriteFileTool",
        func=langchain.tools.file_management.write.WriteFileTool.run,
        description="Write content to a file"
    ),
    Tool(
        name="ClickTool",
        func=langchain.tools.playwright.click.ClickTool.run,
        description="Click on an element with a given CSS selector"
    ),
    Tool(
        name="VectorStoreQATool",
        func=langchain.tools.vectorstore.tool.VectorStoreQATool.run,
        description="Tool for the VectorDBQA chain"
    ),
    Tool(
        name="DeleteFileTool",
        func=langchain.tools.file_management.delete.DeleteFileTool.run,
        description="Delete specified files"
    ),
    Tool(
        name="ExtractTextTool",
        func=langchain.tools.playwright.extract_text.ExtractTextTool.run,
        description="Extracts text from the specified HTML elements"
    ),
    Tool(
        name="ExtractHyperlinksTool",
        func=langchain.tools.playwright.extract_hyperlinks.ExtractHyperlinksTool.run,
        description="Extract hyperlinks from a webpage"
    ),
    Tool(
        name="GetElementsTool",
        func=langchain.tools.playwright.get_elements.GetElementsTool.run,
        description="Gets elements from a webpage using a CSS selector"
    ),
    Tool(
        name="NavigateBackTool",
        func=langchain.tools.playwright.navigate_back.NavigateBackTool.run,
        description="Navigates back to the previous webpage"
    )
]

# Set up the Agent
# Combine everything to set up the agent.

# Define the list of tool names
tools_names = [
    "MoveFileTool",
    "FileSearchTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "WikipediaQueryRun",
    "ArxivQueryRun",
    "CopyFileTool",
    "DuckDuckGoSearchResults",
    "JsonListKeysTool",
    "WriteFileTool",
    "ClickTool",
    "VectorStoreQATool",
    "DeleteFileTool",
    "ExtractTextTool",
    "ExtractHyperlinksTool",
    "GetElementsTool",
    "NavigateBackTool"
]

# Create an instance of LLMSingleActionAgent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=['TASK COMPLETED SUCCESSFULLY, BOSS'],
    allowed_tools=tools
)

stop_sequence = ["\nObservation:"]  # Example stop sequence
allowed_tools = tools_names
# Example allowed tools

llm_single_action_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=stop_sequence,
    allowed_tools=allowed_tools
)

## Initialize MEMORY
###########################################################
# # Initialize FileChatMessageHistory with the path to the file where chat history will be stored
file_chat_history = FileChatMessageHistory(file_path="D:\\PROJECTS\\AGENT_X\\customagentX_chat_history.json")
# Create ConversationBufferMemory with FileChatMessageHistory as vector store
memory_file = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=file_chat_history, input_key="input"
)
# Create ConversationBufferMemory for CombinedMemory
conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines", input_key="input"
)
# Create ConversationSummaryMemory for CombinedMemory
summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
# Combine memories for CombinedMemory
memory_combined = CombinedMemory(memories=[memory_file, conv_memory, summary_memory])
# Create a new instance of AgentExecutor with combined memory
agent_executor_with_memory = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory_combined)
import threading
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import json




user_input = "Create an application that calculates a user's Body Mass Index. Generate the code in python"




class CustomAgent(LLMSingleActionAgent):
    custom_name: str = Field(..., description="Custom name for the agent")
    prompt_template: str = Field(..., description="Prompt template for the agent")
    system_message: str = Field(..., description="System message for the agent")

    class Config:
        extra = "allow"
        
        
# Initialize a dictionary to save input and output
memory_file = {'input': [], 'output': []}

# Common variables (llm, output_parser, and tools need to be initialized before this code)
stop = ["\nObservation:"]
tool_names = [tool.name for tool in tools]


# Global variable to hold summaries from each agent
agent_summaries = []





#########################
# Task List Generator Agent Setup
# Improved Task List Generator Agent Setup
task_list_generator_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="You are the Task List Generator Agent. Analyze the following user input and break it down into a detailed list of tasks, separated by commas, that need to be executed for successful project completion. They must be concise but well thought out tasks to complete the overall objective stated from the user input. User Input: {user_input}"
)

task_list_generator_llm_chain = LLMChain(llm=llm, prompt=task_list_generator_prompt)
task_list_generator_agent = LLMSingleActionAgent(
    llm_chain=task_list_generator_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)

task_list_generator_agent_executor = AgentExecutor.from_agent_and_tools(agent=task_list_generator_agent, tools=tools, verbose=True)


#########################
# Director Agent Setup
# Director Agent Setup
director_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="Your focus is now expert strategic execution and monitoring. Your primary function is to take the task list generated by the Task List Generator Agent and oversee its execution through various agents. After each agent performs its function, summarize their output and decide the next course of action. {user_input}"
)
director_llm_chain = LLMChain(llm=llm, prompt=director_prompt)
director_agent = LLMSingleActionAgent(
    llm_chain=director_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)
director_agent_executor = AgentExecutor.from_agent_and_tools(agent=director_agent, tools=tools, verbose=True)


#########################
# Software Engineer Agent Setup
software_engineer_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="As the Software Engineer Agent, you are working on a specific task as part of a larger operation being orchestrated by the Director Agent. Your responsibilities include crafting efficient algorithms and writing effective code based on the current task. You always used advanced and expert best practices. {user_input}"
)
software_engineer_llm_chain = LLMChain(llm=llm, prompt=software_engineer_prompt)
software_engineer_agent = LLMSingleActionAgent(
    llm_chain=software_engineer_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)
software_engineer_agent_executor = AgentExecutor.from_agent_and_tools(agent=software_engineer_agent, tools=tools, verbose=True)


#########################
# Critic Agent Setup
critic_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="As the Critic Agent, you are part of a larger operation orchestrated by the Director Agent. Your role is to provide critical evaluation of the decisions and work produced by the other agents, focusing on efficiency, effectiveness, and adherence to best practices. Offer constructive criticism and recommend alternatives based on the current task. You always used advanced and expert best practices. {user_input}"
)
critic_llm_chain = LLMChain(llm=llm, prompt=critic_prompt)
critic_agent = LLMSingleActionAgent(
    llm_chain=critic_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)
critic_agent_executor = AgentExecutor.from_agent_and_tools(agent=critic_agent, tools=tools, verbose=True)


#########################
# Tools Agent Setup
tools_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="As the Tools Agent, you are part of a larger operation orchestrated by the Director Agent. Your function is to execute specific tools and utilities required for the completion of the current task. You always used advanced and expert best practices. {user_input}"
)
tools_llm_chain = LLMChain(llm=llm, prompt=tools_prompt)
tools_agent = LLMSingleActionAgent(
    llm_chain=tools_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)
tools_agent_executor = AgentExecutor.from_agent_and_tools(agent=tools_agent, tools=tools, verbose=True)


#########################
# Debugger Agent Setup
debugger_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="As the Debugger Agent, you are part of a larger operation orchestrated by the Director Agent. Your primary objective is to identify issues, bugs, or inefficiencies within the current task and offer effective and minimally disruptive solutions. You always used advanced and expert best practices. {user_input}"
)
debugger_llm_chain = LLMChain(llm=llm, prompt=debugger_prompt)
debugger_agent = LLMSingleActionAgent(
    llm_chain=debugger_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)
debugger_agent_executor = AgentExecutor.from_agent_and_tools(agent=debugger_agent, tools=tools, verbose=True)


#########################
# Architect Agent Setup
architect_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="As the Architect Agent, you are part of a larger operation orchestrated by the Director Agent. Your role is to provide the design and structure for the current task, ensuring that it is functional, scalable, and maintainable. You always used advanced and expert best practices. {user_input}"
)
architect_llm_chain = LLMChain(llm=llm, prompt=architect_prompt)
architect_agent = LLMSingleActionAgent(
    llm_chain=architect_llm_chain,
    output_parser=output_parser,
    stop=stop,
    allowed_tools=tool_names
)
architect_agent_executor = AgentExecutor.from_agent_and_tools(agent=architect_agent, tools=tools, verbose=True)


# Modified Director Agent to handle summarization and forwarding
def director_agent_function(previous_agent_output, current_task):
    # Code to summarize the previous agent's output and decide the next course of action
    summary = f"Summary for {current_task}: {previous_agent_output}"
    # Append the summary to the global list
    agent_summaries.append(summary)
    return summary


    # Run Task List Generator Agent if task list is empty
# Function to get user input or default after a timeout
def get_input(prompt, timeout=20):
    print(prompt)
    timer = threading.Timer(timeout, print, ["User did not respond, defaulting option..."])
    timer.start()
    choice = input()
    timer.cancel()
    return choice

director_agent_input = {}

# Modified run_all_agents_in_sequence function
def run_all_agents_in_sequence(initial_user_input, task_list=None):
    global director_agent_input
    previous_agent_output = ""
    previous_outputs = ""
    final_consensus = ""  # Initialize final_consensus here
    task_list = task_list or []
    
    
    if not task_list:
        print("Running Task List Generator Agent...")
        task_list_str = task_list_generator_agent_executor.run(initial_user_input)
        task_list = task_list_str.split(",")  # Assuming tasks are comma-separated

        while True:
            print(f"Proposed Task List: {task_list_str}")
            print("Choose an option:")
            print("1: Accept task list")
            print("2: Manually redefine tasks")
            print("3: Generate new task list")
            
            choice = get_input("Your choice:", 20)

            if choice == "1":
                break
            elif choice == "2":
                new_task_list_str = get_input("Enter the new task list (comma-separated):")
                task_list = new_task_list_str.split(",") if new_task_list_str else task_list
                break
            elif choice == "3":
                task_list_str = task_list_generator_agent_executor.run(initial_user_input)
                task_list = task_list_str.split(",")
            else:
                print("Defaulting to accepting the task list.")
                break

    next_task = task_list.pop(0) if task_list else None

    if next_task:
        print("Running Director Agent...")
        
        # If director_agent_input is not a dict, initialize it
        if not isinstance(director_agent_input, dict):
            director_agent_input = {}
        
        # Package task_list and previous_outputs into a single JSON-like string
        task_list_str = ",".join(task_list)
        director_agent_input.update({
            "task_list": task_list_str,
            "previous_agent_summary": previous_outputs
        })
        
        # Run the Director Agent with the packaged input
        director_agent_consensus = director_agent_executor.run(json.dumps(director_agent_input))
        previous_outputs += f"\n{director_agent_consensus}"
        
        for task in task_list:
            print(f"Executing task: {task}")

            # Run Software Engineer Agent
            print("Running Software Engineer Agent...")
            software_engineer_output = software_engineer_agent_executor.run(initial_user_input + director_agent_consensus)
            previous_outputs += f"\nSoftware Engineer: {software_engineer_output}"

            # Run Critic Agent
            print("Running Critic Agent...")
            critic_agent_consensus = critic_agent_executor.run(initial_user_input + software_engineer_output + director_agent_consensus)
            previous_outputs += f"\nCritic: {critic_agent_consensus}"    
            
            # Run Tools Agent
            print("Running Tools Agent...")
            tools_agent_output = tools_agent_executor.run(initial_user_input + software_engineer_output + director_agent_consensus + critic_agent_consensus)
            previous_outputs += f"\nTools Agent: {tools_agent_output}"

            # Run Debugger Agent
            print("Running Debugger Agent...")
            debugger_agent_output = debugger_agent_executor.run(initial_user_input + software_engineer_output + director_agent_consensus + critic_agent_consensus + tools_agent_output)
            previous_outputs += f"\nDebugger: {debugger_agent_output}"

            # Run Architect Agent
            print("Running Architect Agent...")
            architect_agent_output = architect_agent_executor.run(initial_user_input + software_engineer_output + director_agent_consensus + critic_agent_consensus + tools_agent_output + debugger_agent_output)
            previous_outputs += f"\nArchitect: {architect_agent_output}"

            # Run Director Agent to summarize and forward
            director_agent_output = director_agent_function(previous_agent_output, task)
            
            # Update previous_agent_output for the next iteration
            previous_agent_output = director_agent_output
            
        # Final consensus by Director Agent
        final_consensus = "Final Director Summary: " + " | ".join(agent_summaries)
        print(final_consensus)
                
        # Continue with the next task
        run_all_agents_in_sequence(initial_user_input, task_list)

    else:
        print("All tasks completed.")



        # Store conversation history
        memory_file['input'].append(initial_user_input)
        memory_file['output'].append(final_consensus)  # Replace final_director_output with final_consensus
    
    
    # Check for task completion or continue task
    if "TASK COMPLETED SUCCESSFULLY, BOSS" in final_consensus:  # Replace final_director_output with final_consensus
        print("Task completed successfully.")
    else:
        # Ask for new user input
        new_input = get_input("Enter new input within 20 seconds or the script will continue: ")
        if new_input:
            run_all_agents_in_sequence(new_input, llm_chain, output_parser, tools)
        else:
            run_all_agents_in_sequence(final_director_output, llm_chain, output_parser, tools)  # Continue along the task route

# Initialize the run with an example input
run_all_agents_in_sequence(user_input)

