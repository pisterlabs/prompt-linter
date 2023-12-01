#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

'''
利用chatgpt，将自然语言自动生成sql，查询数据库，并给出自然语言结果
pip install langchain-experimental
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0, verbose=True)

    # 连接到FlowerShop数据库（之前我们使用的是Chinook.db）
    db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

    # 创建SQL数据库链实例，它允许我们使用LLM来查询SQL数据库
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

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
