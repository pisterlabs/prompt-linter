from langchain import LLMChain
from prompt.translator.translate_text import TRAMSLATOR_PROMPT

async def translate_text(llm):
    chain = LLMChain(llm = llm, prompt = TRAMSLATOR_PROMPT)
    return chain