from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from googlesearch_py import search
from pydantic import Field
import asyncio, time, json
# RemoteTools.py
from RemoteTools import REMOTE_TOOLS

class GoogleSearchTool(BaseTool):
    name = 'GoogleSearch'
    description = (
        'Useful for searching the internet.'
        'Useful for getting current information to provide up-to-date answers.'
        'Do not use this tool to look for phone numbers unless google is specifically mentioned.'
        'Accepts a single argument, which is a string representing the search query.'
        'Returns a list of dictionaries, containing the keys "title", "url", and "description".'
    )

    def _run(self, query):
        return list(search(query))
    
    async def _arun(self, query):
        return self._run(query)

class PlanTool(BaseTool):
    name = "PlanTool"
    description = (
        "Useful for determining a plan of steps to take to achieve a goal."
        "Use this when asked to preform a series of actions."
        "Use this when you need to use a tool with an argument that depends on the output of another (or more) tool(s)."
        "Accepts a single argument, which is a string representing the goal in natural language (be descriptive)."
        "Returns a list of dictionaries, each containing a 'name' and 'argument' key."
        "After using this tool to determine a plan, you will always use the tools in the order they are listed in the plan (filling in placeholders as needed). You do not need to ask before continuing."
    )
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    def _run(self, goal):
        # should i tell it about the PlanTool or is that asking for recursive API bills?
        resp = self.llm.generate([f"""
        You are given a goal: {goal}.
        You must plan out a series of steps to achieve this goal.
        Here is a list of tools you have available to accomplish this goal:
        {', '.join([f'{tool.name}: {tool.description}' for tool in [*REMOTE_TOOLS, *UTIL_TOOLS]])}

        Return a valid JSON object contaning the plan, a list of dictionaries, each containing a 'name' and 'argument' key.
        """])
        res = resp.generations[0][0].text
        try:
            return json.loads(res)
        except json.JSONDecodeError:
            # use a third LLM to fix JSON format? fuck
            return res

    async def _arun(self, goal):
        return self._run(goal)

class SleepTool(BaseTool):
    name = 'Sleep'
    description = (
        'Useful for waiting for a certain amount of time before continuing.'
        'Use this when asked to wait between commands.'
        'Accepts a single argument, which is an integer representing the number of seconds to wait.'
    )

    def _run(self, seconds):
        time.sleep(int(seconds))
        return 'done'

    async def _arun(self, seconds):
        await asyncio.sleep(int(seconds))
        return 'done'

UTIL_TOOLS = [
    PlanTool(),
    SleepTool(),
    GoogleSearchTool(),
]
