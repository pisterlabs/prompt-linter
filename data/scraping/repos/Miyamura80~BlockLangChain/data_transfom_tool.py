from tools.util_tools import SyncTool
from langchain.llms import OpenAI


class DataTransformTool(SyncTool):
    name = "DATA_TRANSFORM_TOOL"
    description = (
        "This is a tool that transforms and data. We consider data extraction as transformation too."
        "You shouldn't use this tool often. It just calls an LLM on the data (previous observation), and the thought "
        "of the core LLM.\n"
        "Example #1: Previous question: Extract the symbol. Observation: {symbol=DAI, "
        "address=0x6B175474E89094C44Da98b954EedeAC495271d0F, decimals=18}. Thought: We have the data that we need, "
        "extract it. Result: DAI."
    )

    def _run(self, query: str) -> str:
        llm = OpenAI(temperature=0, max_tokens=100)
        return llm(f"Transform the data below \n {query}")
