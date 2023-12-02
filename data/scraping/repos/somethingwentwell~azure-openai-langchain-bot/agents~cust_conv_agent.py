from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser, ConversationalChatAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re

# PREFIX = """

# Assistant is a large language model trained by OpenAI.

# Assistant can ONLY answer GitLab related questions.

# If there's any questions are not related to GitLab, reply "I don't know".

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."""

PREFIX = """
You are GitLab Assistant.

You CANNOT answer questions which are not related to GitLab.

If there's any questions are not related to GitLab, you must reply "I don't know".

"""

SUFFIX = """TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. 

The tools the you can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action):

{{{{input}}}}
"""

class CustConvAgent():
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
        super().__init__()

    def setup(self):
        prompt = ConversationalChatAgent.create_prompt(
            tools=self.tools,
            system_message=PREFIX,
            human_message=SUFFIX,
            input_variables=["input", "agent_scratchpad", "chat_history"])
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.tools]
        agent = ConversationalChatAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        return agent