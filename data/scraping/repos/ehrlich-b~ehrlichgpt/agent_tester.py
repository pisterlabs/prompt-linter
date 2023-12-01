from typing import cast
from langchain.llms.base import BaseLLM
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, LLMMathChain, SerpAPIWrapper
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.schema import (HumanMessage, AIMessage)

llm = ChatOpenAI(temperature=0.0) # type: ignore

template="""You are an information retrieval bot, you are given a discord chat conversation, and a set of tools. It is your job to select the proper information collection tools to respond to the last message.

Your (the AI's) discord name is: {discord_name}

Tools format:
Tool['parameter']: Tool description (tools can be called multiple times with different parameters, 0-1 parameter per call)

Tools:
SummarizedMemory[]: Summarized short term conversational memory (last 15-20 messages)
LongTermMemory["embedding_query"]: Long term memory, parameter is the query to use, this will generate a query embedding and search for similar messages from chat history beyond the short term memory
WebSearch["search_query"]: Search the web for fact based information that you don't know (i.e. because it's too recent), but not for general tips you can already make suggestions on unless specifically asked for a web search
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
{message}
"""

# Make a basic hello world prompt
prompt = PromptTemplate(
    template=template,
    input_variables=["message"],
)

chain = LLMChain(llm=llm, prompt=prompt)
print (chain.run(message="adotout#1452: Ok, given everything that's been said, what do you think is the most logical conclusion?"))
