from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
template = '''
Question: {query}. Please response on chinese with markdown.

Anwser: let's think step by step.
'''
prompt = PromptTemplate(template=template, input_variables=["query"])
llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI())
llm_chain.run('假设有一个池塘，里面有无穷多的水。现有2个空水壶，容积分别为5升和6升。问题是如何只用这2个水壶从池塘里取得3升的水?')