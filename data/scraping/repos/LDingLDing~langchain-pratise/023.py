from dotenv import load_dotenv
load_dotenv()

from langchain.utilities import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

llm = OpenAI(temperature=0, verbose=True)

db_chain = SQLDatabaseChain.from_llm(llm,db,verbose=True)


# 运行与鲜花运营相关的问题
response = db_chain.run("有多少种不同的鲜花？")
print(response)

response = db_chain.run("哪种鲜花的存货数量最少？")
print(response)

response = db_chain.run("平均销售价格是多少？")
print(response)

response = db_chain.run("从法国进口的鲜花有多少种？")
print(response)

response = db_chain.run("哪种鲜花的销售量最高？")
print(response)