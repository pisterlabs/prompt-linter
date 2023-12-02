from langchain import OpenAI
from langchain.tools import StructuredTool


def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    return a * b


llm = OpenAI(temperature=0)
tool = StructuredTool.from_function(multiplier)
