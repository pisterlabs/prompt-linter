import os, subprocess, socket
from sys import platform
from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

'''
TODO

    - Add usage (https://python.langchain.com/docs/modules/model_io/models/llms/how_to/token_usage_tracking) -- maybe add tool for showing usage when prompted? 
        could do a flow like --- "User: -u" , "Chatbot: Tokens, Cost($), etc."
    - Configure Autorun functionality to bypss Y/n sequence
    - Configure conversation-loading (ability to load conversation from file 
    - add more init parameters to make more configurable (verbose to log thought process [now false by default]
            max_iterations to limit number of iterations [now 5 by default], memory to load conversation from file [now None by default])

'''

class ShellSensei:
    price = 0.0
    coins = 0
    hostName = socket.gethostname()
    def __init__(self, model="gpt-4", temperature=0.5, verbose=False):
        load_dotenv(find_dotenv())
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.verbose = verbose
        self.term_gpt = ChatOpenAI(model=model, temperature=temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
        self.system_message = open("HyperParams/system_message.txt", "r").read()
        self.command_executor = self._init_command_executor()
        self.tools = [self.command_executor]
        self.template = self._init_template()
        self.agent = self._init_agent()

    def _run_command(self, text):
        try:
            command_output = subprocess.check_output(text, shell=True, stderr=subprocess.STDOUT)
            return command_output
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output.decode('utf-8')}"

    def _init_command_executor(self):
        description = (
            "This method is used to run a command in the terminal. "
            "It takes in one str param which is the command to be run, "
            "runs it, and returns output if no error occurs, else, it returns the error message."
        )
        return Tool(name="command-executor", func=self._run_command, description=description)

    def _init_template(self):
        template = (
            f"{self.system_message}\n\nYou are running on {{platform}}. "
            "\n\nHuman: {human_input}\nChatbot:"
        )
        return PromptTemplate(input_variables=["human_input", "platform"], template=template)

    def _init_agent(self):
        return initialize_agent(
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            tools=self.tools,
            llm=self.term_gpt,
            verbose=self.verbose,
            max_iterations=5,
            memory=self.memory
        )

    def interact(self):
        while True:
            query = input(f"{ShellSensei.hostName}: ")
            with get_openai_callback() as cb:
                if query.__contains__('cost'):
                    print("Total Price: " + str(ShellSensei.price))
                elif query.__contains__('tokens'):
                    print("Total Tokens: " + str(ShellSensei.coins))
                else :
                    print(self.agent(self.template.format(human_input=query, platform=platform))["output"])
                    ShellSensei.coins += cb.prompt_tokens
                    ShellSensei.price += cb.total_cost

