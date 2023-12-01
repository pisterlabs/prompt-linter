import os

from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationEntityMemory

from semterm.agent.TerminalAgentPrompt import PREFIX
from semterm.config.Config import Config
from semterm.agent.TerminalAgent import TerminalAgent
from semterm.agent.TerminalAgentExecutor import TerminalAgentExecutor
from semterm.terminal.TerminalOutputParser import TerminalOutputParser
from semterm.terminal.TerminalTool import TerminalTool
from semterm.terminal.SemanticTerminalManager import SemanticTerminalManager
from semterm.langchain_extensions.tools import MistakeTool, TerminalHumanTool


class MrklAgent:
    def __init__(self, config: Config):
        config = config.get()
        self.verbose = config.getboolean("DEFAULT", "verbose")
        self.max_iterations = config.getint("DEFAULT", "max_iterations")
        self.timeout = config.getint("DEFAULT", "timeout")
        self.print_terminal_output = config.getboolean(
            "DEFAULT", "print_terminal_output"
        )

        self.llm = ChatOpenAI(temperature=0)
        self.tools = self.load_tools()
        self.memory = self.initialize_memory()

        self.terminal_agent = self.initialize_agent()
        self.terminal_agent_executor = self.initialize_executor()

    def load_tools(self):
        tools = [
            TerminalTool(manager=SemanticTerminalManager()),
            TerminalHumanTool(),
            MistakeTool(),
        ]

        return tools

    def initialize_memory(self):
        return ConversationEntityMemory(
            llm=self.llm,
            return_messages=True,
            chat_history_key="chat_history",
        )

    def initialize_agent(self):
        return TerminalAgent.from_llm_and_tools(
            self.llm,
            self.tools,
            memory=self.memory,
            system_message=PREFIX.format(current_directory=os.getcwd()),
            output_parser=TerminalOutputParser(),
            verbose=self.verbose,
        )

    def initialize_executor(self):
        return TerminalAgentExecutor.from_agent_and_tools(
            self.terminal_agent,
            self.tools,
            memory=self.memory,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )

    def run(self, user_input):
        return self.terminal_agent_executor.run(input=user_input)
