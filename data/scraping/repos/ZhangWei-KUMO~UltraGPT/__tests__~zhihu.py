import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('BACKEND_OPENAI_API_KEY')
template = """在 以下是网站的返回的内容
>>> {requests_result} <<<
请以markdown的格式输出
"""
prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=OPENAI_API_KEY)
llm_chain=LLMChain(llm=llm, prompt=prompt)
chain = LLMRequestsChain(llm_chain=llm_chain)

response = chain({ "url": "https://bigquant.com/docs/devdata.html"})
# 返回的对象中包括output和url两个字段
