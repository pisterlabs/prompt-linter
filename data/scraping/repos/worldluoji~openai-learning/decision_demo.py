
import openai, os

openai.api_key = os.environ.get("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains import LLMChain

llm = OpenAIChat(max_tokens=2048, temperature=0.5)
multiple_choice = """
请针对 >>> 和 <<< 中间的用户问题，选择一个合适的工具去回答她的问题。只要用A、B、C的选项字母告诉我答案。
如果你觉得都不合适，就选D。

>>>{question}<<<

我们有的工具包括：
A. 一个能够查询商品信息，为用户进行商品导购的工具
B. 一个能够查询订单信息，获得最新的订单情况的工具
C. 一个能够搜索商家的退换货政策、运费、物流时长、支付渠道、覆盖国家的工具
D. 都不合适
"""
multiple_choice_prompt = PromptTemplate(template=multiple_choice, input_variables=["question"])
choice_chain = LLMChain(llm=llm, prompt=multiple_choice_prompt, output_key="answer")

question = "请问你们的货，能送到三亚吗？大概需要几天？"
print(choice_chain(question))

question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
print(choice_chain(question))