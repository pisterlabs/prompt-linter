import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('BACKEND_OPENAI_API_KEY')
template = """在 以下是东方财富返回的内容
Extract the answer to the question '{query}' or say "没有发现相关信息" if the information is not contained.
Use the format
<answer>
>>> {requests_result} <<<
请以markdown的格式输出
"""
prompt = PromptTemplate(
    input_variables=["query","requests_result"],
    template=template
)

char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
# docs = char_text_splitter.split_documents(prompt)


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=OPENAI_API_KEY)
llm_chain=LLMChain(llm=llm, prompt=prompt)

chain = LLMRequestsChain(llm_chain=llm_chain)
question = "请下面的内容进行总结"
inputs = {
    "query": question,
    "url": "https://finance.eastmoney.com/"
}
# response = chain(inputs)
# # 返回的对象中包括output和url两个字段
# print(response['output'])