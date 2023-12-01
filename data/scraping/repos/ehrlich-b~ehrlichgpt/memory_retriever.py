from builtins import str
import asyncio
import datetime
import re
from typing import Callable, List, Tuple
from langchain.llms.base import BaseLLM
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, LLMMathChain, SerpAPIWrapper
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.schema import (HumanMessage, AIMessage)
from utils import get_formatted_date


class MemoryRetriever:
    SUMMARIZED_MEMORY = "SummarizedMemory"
    LONG_TERM_MEMORY = "LongTermMemory"
    WEB_SEARCH = "WebSearch"
    TEMPLATE = """You are an information retrieval bot, you are given a discord chat conversation, and a set of tools. It is your job to select the proper information collection tools to respond to the last message.

Current date: {current_date}
Your (the AI's) discord name is: {discord_name}

Tools format:
Tool['parameter']: Tool description (tools can be called multiple times with different parameters, 0-1 parameter per call)

Tools:
SummarizedMemory[]: Summarized short term conversational memory (last 15-20 messages)
LongTermMemory["embedding_query"]: Long term memory, parameter is the query to use, this will generate a query embedding and search for similar messages from chat history beyond the short term memory
WebSearch["search_query"]: Search the web for fact based information that you don't know (i.e. because it's too recent)
CodeGen[]: If the user is making a request for any type of code generation, you must call this tool
Answer[]: You've triggered collection of all the information the answer synthesizer bot will need

Example 1:
sara#7890: Can you remind me what we discussed yesterday about the meeting agenda?
Thought: This message is asking for a summary of a previous conversation from beyond the short-term memory
Tools:
LongTermMemory["sara#7890 meeting agenda"]
SummarizedMemory[]
Answer[]

Example 2:
jane#5678: What's the latest news on Mars exploration?
Thought: This message requires recent information which is not present in the chat history
Tools:
WebSearch["latest news Mars exploration"]
Answer[]

Example 3:
peter#1234: I remember we talked about the benefits of a keto diet and the side effects of intermittent fasting. Can you give me a quick summary?
Thought: This message requires information from two separate previous conversations
Tools:
LongTermMemory["keto diet benefits"]
LongTermMemory["intermittent fasting side effects"]
Answer[]

END EXAMPLES
{message}"""

    def __init__(self) -> None:
        llm = ChatOpenAI(temperature=0.0) # type: ignore

        prompt = PromptTemplate(
            template=self.TEMPLATE,
            input_variables=["message", "discord_name", "current_date"],
        )

        self.chain = LLMChain(llm=llm, prompt=prompt)

    def _parse_tools(self, output: str) -> List[Tuple[str, str]]:
        print(output)
        try:
            tools_section = re.search(r'Tools:\n(.*)', output, re.DOTALL)
            if tools_section:
                tools = tools_section.group(1).strip().split('\n')
                parsed_tools = []
                for tool in tools:
                    try:
                        tool_name, param = tool.strip().split('[')
                        param = param[:-1]
                        parsed_tools.append((tool_name, param.strip('"')))
                    except:
                        print("Tool request malformed: " + tool)
                return parsed_tools
        except:
            pass
        return []


    def run(self, message: str) -> List[Tuple[str, str]]:
        output = self.chain.run(message=message)
        return self._parse_tools(output)

    async def arun(self, message: str, discord_name: str) -> List[Tuple[str, str]]:
        output = await self.chain.arun(message=message, discord_name=discord_name, current_date=get_formatted_date())
        return self._parse_tools(output)
