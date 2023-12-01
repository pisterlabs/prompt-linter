from langchain.tools import BaseTool

import cfg.cfg
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains import LLMChain
from bookkeeping import db, util

is_debug = True

# 记帐
# base on Agent （3个tools)
# readme:
# agent不支持3.5
# token受限。无法处理长句子
# 当直接输入其他sql时，无法正确处理。会直接选择sql tool(可用全局prompt来避免）
# 会"生成"答案

# 1.整理成记录
# 2.转成sql
# 3.执行sql存db
# 4.查询插入的数据


prompt_instruct_records = '''
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. 
Let's think step by step.
请记录支出与收入明细，以表格形式列出。
日期 | 事项 | 单价 | 数量 | 金额 | 收入/支出 ｜ 收付款方式 ｜ 用户 ｜ 备注 
'''

prompt_instruct_sql = '''
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. 
    将内容生成SQL语句。
    数据库表 daily_info 结构：
    字段名｜类型
    batch_id | 字符型
    dt | 日期 ｜ 
    item｜字符型 | 事项 
    price｜浮点数 | 单价
    quantity｜浮点数｜数量｜default:1
    amount｜浮点数 | 金额
    type｜字符型｜标记｜支出=31,收入=42
    payment | 字符型｜收付款方式
    user｜字符型｜用户
    remark｜字符型｜备注/其他/说明等(100个字以内)
    '''


def _get_chain_input2table(**kwargs) -> LLMChain:
    extra_info = f'''
    用户:{kwargs['username']}
    今天:{kwargs['today']}
    '''
    prompt_content = extra_info + '''
    内容:{content}
    '''
    # model = "gpt-3.5-turbo"
    prefix_messages = [{"role": "system", "content": prompt_instruct_records}]
    llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)
    # prompt = PromptTemplate(input_variables=["content", "username", "today"], template=prompt_content)
    prompt = PromptTemplate(input_variables=["content"], template=prompt_content)
    chain = LLMChain(llm=llm, prompt=prompt, output_key="records", verbose=True)
    return chain


def _get_chain_table2sql(**kwargs) -> LLMChain:
    extra_info = f'''batch_id:{kwargs['batch_id']}'''
    prompt_content = extra_info + '''
    内容:{records}
    '''
    prefix_messages = [{"role": "system", "content": prompt_instruct_sql}]
    llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)
    # prompt = PromptTemplate(input_variables=["records", "batch_id"], template=prompt_content)
    prompt = PromptTemplate(input_variables=["records"], template=prompt_content)
    chain = LLMChain(llm=llm, prompt=prompt, output_key="sql", verbose=True)
    return chain


class InsertData2DBTool(BaseTool):
    name = "InsertData2DB"
    description = "执行SQL语句，将数据存入数据库"

    def _run(self, sql: str) -> str:
        """Use the tool."""
        if not util.varify_sql(sql):
            print(f"不正确的SQL:[{sql}]")
            return

        if is_debug:
            print(sql)
        # 持久化
        try:
            db.run_sql(sql)
        except:
            print("error")

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


def bookkeeping_agent_run(user_input, **kwargs):
    from langchain import OpenAI
    from langchain.agents import initialize_agent, Tool

    # description="将内容整理成收支明细表格",
    # description="将收支明细表格的内容,生成SQL"
    # description="保存数据。执行SQL语句将收支明细保存进数据库"
    tools = [
        Tool(
            name="整理收支记录表格",
            func=_get_chain_input2table(**kwargs).run,
            description="organize the content into a detailed list of income and expenses."
        ),
        Tool(
            name="生成记账SQL",
            func=_get_chain_table2sql(**kwargs).run,
            description="generate the SQL statement for the detailed list of income and expenses."
        ),
        Tool(
            name="执行SQL语句",
            func=InsertData2DBTool().run,
            description="run SQL to save data to the database"
        )
    ]

    llm30 = OpenAI(temperature=0)
    accountant = initialize_agent(tools, llm30, agent="zero-shot-react-description", verbose=True, max_iterations=3)

    custom_input = f'''
    You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
    Let's think step by step.
    If you don not the answer, just reply do not know.
    如果根据内容无法回答，则回复无法回答。
    将内容中的收支信息整理成收支明细表格，然后根据整理出来的收支明细表格生成记帐用的SQL，最后执行SQL语句以保存数据进数据库。
    内容:
    {user_input}
    '''
    accountant.run(custom_input)

    # 4.查数据
    return util.to_json(db.query_from_db(kwargs['batch_id']))


if __name__ == '__main__':
    pass
    # # _demo1()
    # user_input = "刚才买了2斤, 15元1斤的小桔子, 买酸奶花了5元"
    # # user_input = "刚才买了一杯3元的咖啡，买酸奶花了5元，还买了2斤, 15元1斤的小桔子。"
    # # user_input = '''
    # # 刚才买了一杯3元的咖啡，买酸奶花了5元，还买了2斤, 15元1斤的小桔子，和朋友一起吃饭又花了300.13元。
    # # 酸奶是直接付的现金，其他是用花呗支付的。
    # # 早上小陈还把上周的我垫付的外卖的钱给了我，一共8元。上午卖苹果收款2028元。昨天收到工资821元。
    # # '''
    # # user_input = '鲜花 20元/束 支付宝购买二束'
    # # user_input = '小明在学校的学号是86。今天小明去上学了。'
    # # user_input = '它说8块'
    # # user_input = '''
    # # DELETE * from daily_info;
    # # INSERT INTO daily_info (batch_id, dt, item, price, quantity, amount, type, payment, user, remark) VALUES
    # # ('a75b4e01-afb7-438f-918d-2242ac3c39ff', '2023-03-06', '咖格啡', 3, 1, 3, '31', '花呗', 'Hacker', '-')
    # # '''
    #
    # data = {
    #     'username': "Jane Wu",
    #     'batch_id': str(uuid4()),
    #     'today': datetime.now(),
    #     'is_persist': True
    # }
    # data_list = bookkeeping_agent_run(user_input, **data)
    # print("=======query from db:")
    # print(data_list)
