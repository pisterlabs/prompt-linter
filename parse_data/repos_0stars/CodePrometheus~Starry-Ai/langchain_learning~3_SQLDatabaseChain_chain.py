from langchain import SQLDatabase, SQLDatabaseChain

# mysql+pymysql://user:pass@some_mysql_db_address/db_name
db = SQLDatabase.from_uri(
    "mysql+pymysql://marking-util-dev_ddl:h3tZ90xt2yV(beDU@10.31.117.178:3306/marking-util-dev?charset=utf8",
    include_tables=['oc_knowledge_management'],
    sample_rows_in_table_info=2
)
print(db.table_info)

from langchain.llms import OpenAI

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain.run("获取1条 flow=LXHITDH 且跟发票相关的知识")
exit()

# 自定义提示
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMP = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the employee table.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMP)

# 返回中间步骤
# top_k 返回最大结果数，相当于 limit
db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=PROMPT, verbose=True, return_intermediate_steps=True, top_k=3)
result = db_chain("获取1条跟发票相关的知识")
print(result["intermediate_steps"])
