from autoagent import (
    Agent, Observation, Thought, ThoughtHandler
)

from functools import partial
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.tools.base import BaseTool
from textwrap import dedent
from typing import Optional

langchain_tool_names = ["serpapi", "python_repl", "wikipedia"]
langchain_tools = load_tools(langchain_tool_names, llm=OpenAI(temperature=0))
default_example_text = [
    """\
        Search: Recursion 

        Observation: Recursion occurs when the definition of a concept or process depends on a simpler version of itself. Recursion is used in a variety of disciplines ranging from linguistics to logic.
    """,
    """\
        Python REPL: print([1, 2, 3] * 3)

        Observation: [1, 2, 3, 1, 2, 3, 1, 2, 3] 
    """,
    """\
        Wikipedia: pi

        Observation: Summary: The number π (; spelled out as "pi") is a mathematical constant that is the ratio of a circle's circumference to its diameter, approximately equal to 3.14159. The number π appears in many formulae across mathematics and physics. It is an irrational number, meaning that it cannot be expressed exactly as a ratio of two integers, although fractions such as {\displaystyle {\tfrac {22}{7}}} are commonly used to approximate it. Consequently, its decimal representation never ends, nor enters a permanently repeating pattern. It is a transcendental number, meaning that it cannot be a solution of an equation involving only sums, products, powers, and integers. The transcendence of π implies that it is impossible to solve the ancient challenge of squaring the circle with a compass and straightedge. The decimal digits of π appear to be randomly distributed, but no proof of this conjecture has been found.
    """,
]

class LangChainDoer(ThoughtHandler):
    def __init__(
            self,
            tools: list[BaseTool],
            agent: "Agent",
        ):
        self.tools = tools
        super().__init__(agent)

    @property
    def description(self) -> str:
        return self.tool.description()

    @property
    def init_examples(self) -> list[str]:
        return self._examples

    def handle_thought(self, action: Thought) -> Optional[list[Thought]]:
        if action.label in [t.name for t in self.tools]:
            print("Handling langchain tool action", action.full)
            return Observation.from_full(self.tool.run(action.full))

DEFAULT_DOERS = [
    partial(
        LangChainDoer,
        tools=langchain_tools,
        examples=default_example_text,
    )
]
