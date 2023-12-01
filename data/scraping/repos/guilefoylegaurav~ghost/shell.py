import os

from langchain.chat_models import ChatOpenAI
from langchain.tools import ShellTool
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import find_dotenv, load_dotenv
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import warnings

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())


def _get_agent_chain():
    llm = ChatOpenAI(callbacks=[StreamingStdOutCallbackHandler()], temperature=0.2)
    prefix = f"""Have a conversation with a human, answering the following questions as best you can. Ensure the 
    commands are for the OS {os.environ.get("OS_NAME")} and the shell {os.environ.get("SHELL_NAME")}. """
    suffix = """Begin! {chat_history} Question: {input} {agent_scratchpad}"""

    def _approve(_input: str) -> bool:
        tokens = _input.split(' ')
        if len(tokens) > 0 and tokens[0] == 'echo':
            return True
        msg = (
            "\nLets run this command?"
        )
        msg += "\n\n" + _input + "\n"
        resp = input(msg)
        return resp.lower() in ("yes", "y")

    callbacks = [HumanApprovalCallbackHandler(approve=_approve)]
    shell_tool = ShellTool(callbacks=callbacks, handle_tool_error=True)
    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")
    tools = [shell_tool]
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    return agent_chain


class Shell:
    def __init__(self, response_strategy) -> None:
        self.agent = _get_agent_chain()
        self.response_strategy = response_strategy

    def repl(self):
        self.response_strategy.respond(self.agent)
