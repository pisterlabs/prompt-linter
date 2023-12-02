"""
https://python.langchain.com/docs/modules/chains/foundational/llm_chain

最简单的 LLMChain
"""

import env

from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is a good name for a company that makes {product}?"

llm = OpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
llm_chain("colorful socks")


"""
另外的调用方式

使用 __call__
"""

input_list = [
    {"product": "socks"},
    {"product": "computer"},
    {"product": "shoes"}
]

result = llm_chain.apply(input_list)
print("result:", result)

"""
generate 方法，返回更多信息， such as token usages and finish reason.
"""

result = llm_chain.generate(input_list)
print("generate called result:", result)


""""
predict 
Predict 与 run 方法类似，只不过输入键被指定为关键字参数而不是 Python 字典。
"""

result = llm_chain.predict(product="colorful socks")
print("result called result:", result)

# Multiple inputs example

template = """Tell me a {adjective} joke about {subject}."""
prompt = PromptTemplate(template=template, input_variables=["adjective", "subject"])
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0))

llm_chain.predict(adjective="sad", subject="ducks")


"""
Parsing the outputs
"""

from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
template = """List all the colors in a rainbow"""
prompt = PromptTemplate(template=template, input_variables=[], output_parser=output_parser)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# llm_chain.predict()

result = llm_chain.predict_and_parse()

print("predict_and_parse result:", result)


"""
Initialize from string
You can also construct an LLMChain from a string template directly.
"""

template = """Tell me a {adjective} joke about {subject}."""
llm_chain = LLMChain.from_string(llm=llm, template=template)

llm_chain.predict(adjective="sad", subject="ducks")
