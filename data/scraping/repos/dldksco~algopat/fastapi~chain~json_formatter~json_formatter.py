from langchain import LLMChain
from prompt.json_formatter.json_formatter_prompt import JSON_FORMATTER_PROMPT

async def json_formatter(llm):
    chain = LLMChain(llm = llm, prompt = JSON_FORMATTER_PROMPT)
    return chain