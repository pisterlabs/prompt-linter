from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool, Tool

import charact_selector
import prompt_manages
from LanguageModelSwitcher import LanguageModelSwitcher


# #知识类工具
# def chat():
#     model = LanguageModelSwitcher("minimax").model
#     tuji = charact_selector.initialize_tuji()
#     prompt = prompt_manages.rolePlay() + tuji + prompt_manages.charactorStyle()
#     final_prompt = ChatPromptTemplate.from_template(prompt)
#     # final_prompt.format_prompt(char="兔几", user="大头", input="你好")
#     chain = final_prompt | model
#     for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": "你好"}):
#         return chunk
#     pass


class ChatTool(BaseTool):
    name = "Chat"
    # func = chat()
    description = "Use the ChatTool when the user talks about their daily life, asks common questions, or seeks casual conversation."

    def _run(self, query: str) -> str:
        model = LanguageModelSwitcher("text_gen").model
        tuji = charact_selector.initialize_tuji()
        prompt = prompt_manages.rolePlay() + tuji + prompt_manages.charactorStyle()
        final_prompt = ChatPromptTemplate.from_template(prompt)
        # final_prompt.format_prompt(char="兔几", user="大头", input="你好")
        chain = final_prompt | model
        t =  chain.invoke({"user": "大头", "char": "兔叽", "input": "你好"})
        # for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": "你好"}):
        return t

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")




class MemoryTool(BaseTool):
    name = " Memories"
    description = "Use the MemoryTool when the user inquires about past events, memories, or content of their previous conversations."

    def _run(self, query: str) -> str:
        return "我是记忆思维链"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

class ActionResponeTool(BaseTool):
    name = "Actions"
    description = "Use the ActionResponeTool when the user talks about actions, decisions, or seeks advice related to actions."

    def _run(self, query: str) -> str:
        return "我是游戏行为思维链"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

class GameKnowledgeTool(BaseTool):
    name = "Games or gameplay or rules or tips or game strategies or game updates or video gamesor game characters"
    description = "Use the ActionResponseTool when the user seeks advice on specific actions, discusses upcoming activities, or inquires about how to react"

    def _run(self, query: str) -> str:
        model = LanguageModelSwitcher("qianfan").model
        tuji = charact_selector.initialize_tuji()
        prompt = prompt_manages.rolePlay() + tuji + prompt_manages.charactorStyle() + prompt_manages.plotDevelopment()
        final_prompt = ChatPromptTemplate.from_template(prompt)
        # final_prompt.format_prompt(char="兔几", user="大头", input="你好")
        chain = final_prompt | model
        for chunk in chain.stream({"user": "大头", "char": "兔叽", "input":"你好"}):
            return chunk

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")