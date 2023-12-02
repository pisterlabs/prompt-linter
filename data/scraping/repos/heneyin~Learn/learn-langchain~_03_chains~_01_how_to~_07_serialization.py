"""
https://python.langchain.com/docs/modules/chains/how_to/serialization
本笔记本介绍了如何将链序列化到磁盘和从磁盘序列化。我们使用的序列化格式是json或yaml。

"""

"""
Saving a chain to disk
"""
import env
from langchain import PromptTemplate, OpenAI, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

llm_chain.save("llm_chain.json")

"""
load
"""

from langchain.chains import load_chain

chain = load_chain("llm_chain.json")

chain.run("whats 2 + 2")

