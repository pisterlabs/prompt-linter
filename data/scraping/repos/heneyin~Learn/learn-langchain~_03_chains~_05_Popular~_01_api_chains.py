"""

https://python.langchain.com/docs/modules/chains/popular/api

APIChain 允许使用 LLM 与 API 交互以检索相关信息。通过提供与所提供的 API 文档相关的问题来构建链。

"""

import env

from langchain.chains.api.prompt import API_RESPONSE_PROMPT


from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)


from langchain.chains.api import open_meteo_docs
chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)

chain_new.run('What is the weather like right now in Munich, Germany in degrees Fahrenheit?')
