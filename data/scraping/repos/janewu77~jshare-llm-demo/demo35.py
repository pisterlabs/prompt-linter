from langchain.chains.base import SimpleMemory

import cfg.cfg
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains import LLMChain, SequentialChain
from bookkeeping import db, util

# 记帐 当前效果最佳 base on 3.5

prompt_instruct_records = '''
请为用户整理记帐信息，以表格形式列出。
交易日期 | 事项 | 单价 | 数量(默认1) | 数量单位(默认个) | 金额 | 收入/支出 ｜ 收付款方式(默认现金) ｜ 备注 
'''
# "找零/charge"是指商家退回给顾客的多余金额，通常发生在顾客付款时支付的金额超过了商品或服务的价格。
# 请记录支出与收入明细，以表格形式列出。
# “昨天”是指比今天更早的一天。
# “前天”是指比昨天更早的一天。
# “大前天”是指比前天更早的一天。
# 如果今天是 2020-5-9，则昨天是2020-5-8, 前天是2020-5-7, 大前天是 2020-5-6.

prompt_instruct_sql = '''
数据库表 transaction_info 结构：
字段名｜类型
batch_id | 字符型
transaction_date | 交易日期 ｜ 
item｜字符型 | 事项 
price｜浮点数 | 单价
quantity｜浮点数｜数量｜default:1
quantity_unit｜字符型｜数量单位｜default:个
amount｜浮点数 | 金额
ttype｜字符型｜标记｜支出=31,收入=42
payment | 字符型｜收付款方式
username｜字符型｜用户
remark｜字符型｜备注/其他/说明等(100个字以内)
'''


class Accountant35(object):

    __prompt_system_prefix = '''
    You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. 
    Let's think step by step.
    '''

    def _get_chain_input2table(self) -> LLMChain:
        prompt_system = self.__prompt_system_prefix + prompt_instruct_records

        prompt_content = '''
        今天:{today}
        内容:{content}
        '''
        # model = "gpt-3.5-turbo"
        prefix_messages = [{"role": "system", "content": prompt_system}]
        llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)
        prompt = PromptTemplate(input_variables=["content", "today"], template=prompt_content)
        chain = LLMChain(llm=llm, prompt=prompt, output_key="records", verbose=True)
        return chain

    def _get_chain_table2sql(self) -> LLMChain:
        prompt_other = '''
        将输入的内容转换成SQL的Insert语句。
        仅输出SQL语句。如果无法换成SQL的Insert语句，请回复无法处理。
        '''
        prompt_system = self.__prompt_system_prefix + prompt_instruct_sql + prompt_other
        prompt_content = '''
        用户:{username}
        batch_id:{batch_id}
        内容:{records}
        '''
        prefix_messages = [{"role": "system", "content": prompt_system}]
        llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)
        prompt = PromptTemplate(input_variables=["records", "username", "batch_id"], template=prompt_content)
        chain = LLMChain(llm=llm, prompt=prompt, output_key="res_sql", verbose=True)
        return chain

    def _save2db(self, sql_insert, verbose=True):
        if verbose:
            print('\n\n> Entering custom chain[save2db]...')
            print(f'input sql:[{sql_insert}]')

        db.run_sql(sql_insert, verbose)
        if verbose:
            print('> Finished custom chain[save2db]...\n\n')

    # 根据用户输入，进行记帐，数据进行持久化
    # 整理成记录 > 转成sql > 执行sql存db > 查询插入的数据
    def recording(self, content, **kwargs):
        return self._bookkeeping_seq_chain_run(content, **kwargs)

    def _bookkeeping_seq_chain_run(self, content, **kwargs):
        batch_id = kwargs['batch_id']
        username = kwargs['username']

        # 1.格式化文字
        chain_first = self._get_chain_input2table()
        # 2.生成sql
        chain_second = self._get_chain_table2sql()

        overall_chain = SequentialChain(
            memory=SimpleMemory(memories=kwargs),
            chains=[chain_first, chain_second],
            input_variables=["content"],
            verbose=True)
        res_sql = overall_chain.run(content=content)

        # 3. save to db
        self._save2db(res_sql)

        # 4. query from db
        res = util.to_json(db.query_from_db(batch_id, username))
        return res

    def confirm(self, batch_id, username):
        return db.confirm(batch_id, username)

    def delete(self, batch_id, username, iid):
        return db.confirm(batch_id, username, iid)

    # 用自然语言查询某段时间内的帐目明细、小计
    def talk_over_book(self, username):
        pass


if __name__ == '__main__':
    # _demo1()
    pass
