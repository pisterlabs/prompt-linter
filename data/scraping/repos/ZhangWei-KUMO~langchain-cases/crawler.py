import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
分析网页的HTML内容，我希望知道有哪些影片的名称。把影片名称并翻译成中文
>>> {requests_result} <<<
"""
prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://www.javbus.com/"
}

response = chain(inputs)
# 返回的对象中包括output和url两个字段
print(response['output'])