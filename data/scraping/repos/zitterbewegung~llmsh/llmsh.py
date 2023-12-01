#!/usr/bin/env python3
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import ShellTool
from dotenv import load_dotenv
from prompt_toolkit import prompt
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os, sys

load_dotenv()

shell_tool = ShellTool()
# template = """
# You are an expert in using shell commands. The command will be directly executed in a shell. Only provide a single executable line of shell code as the answer to {question}.
# """

# prompt = PromptTemplate(template=template, input_variables=["question"])

llm = ChatOpenAI(temperature=0)
# llm_chain = LLMChain(prompt=prompt, llm=llm)

if len(sys.argv) > 1:
    if sys.argv[1] == "--l" or sys.argv[1] == "-l" or sys.argv[1] == "-local":
        from local import llm_local

        print("Using local llama")
        llm = llm_local


shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


agent = initialize_agent(
    [shell_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)


def main():
    while True:
        command = input("$ ")  # prompt("$)
        if command == "exit":
            break
        elif command == "help":
            print("llmsh: a simple natural language shell in python.")
        else:
            output = agent.run(command)
            print(output)


if __name__ == "__main__":
    main()
