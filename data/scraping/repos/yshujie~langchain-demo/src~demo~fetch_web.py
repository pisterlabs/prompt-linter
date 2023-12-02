from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
 
# init llm
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
 
# set prompt
template = """在 >>> 和 <<< 之间是网页返回的 HTML 内容。
网页是新浪财经 A 股上市公司的简介。
请抽取参数请求的信息。
 
>>> {requests_result} <<<
 
请使用如下格式的 JSON 格式返回数据
{{
        "company_name": "",
        "company_english_name": "",
        "issue_price": "",
        "date_of_establishment": "",
        "registered_capital": "",
        "office_address": "",
        "company_profile": "",
}}
Extracted:
"""
 
prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)
 
# init chain
chain = LLMRequestsChain(
    llm_chain=LLMChain(llm=llm, prompt=prompt)
)
 
def fetch_web(url):
     inputs = {
         "url": url
     }
     
     response = chain(inputs)
     print(response)