import os
import openai

openai.api_type = "azure"
openai.api_key = "XXXXX"
openai.api_base = "https://openaidemo-hu.openai.azure.com/"
openai.api_version = "2022-12-01"

prompt_prefix = """### MySQL SQL tables, with their properties:
#
# table POFD_BS_SHARENUMLIMITED
# (
#  id int NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT 'Primary Key',
#  QUOTA_ITEM          VARCHAR(200) not null COMMENT '限额项目',
#  INVEST_CHANNEL      VARCHAR(200) COMMENT '投资渠道、销售方式',
#  QUOTA_CONTAIN_ID    VARCHAR(10) COMMENT '限额包含标识、是否含申购费是否包含',
#  QUOTA_NUM           DOUBLE COMMENT  '金额数、限额数',
#  UNIT                VARCHAR(20) COMMENT '金额数、限额数'
# ) COMMENT '数据部门，从pdf文件中抽取数据，作为结构化数据存储在这张表中，供其他部门使用';
#
# 
# 根据上面的表定义，json的数组数据，转成多条记录的insert 语句。
# json数组中，每个大括号{}对应着表中的一条记录。
# 这个json数组中有多少个大括号{}，就生成多少条记录的语句。
# 例如有三个{},就要values后面就需要三条记录对应,对应的输入输出如下：
[
{
        "限额项目": "申购最低额",
        "销售方式": "直销中心柜台",
        "是否含申购费": "含",
        "金额数": "10万",
        "单位": "元"
}
,

{
        "限额项目": "追加申购最低额",
        "销售方式": "直销中心柜台",
        "是否含申购费": "含",
        "金额数": "10万",
        "单位": "元"
}
]
insert into POFD_BS_SHARENUMLIMITED(QUOTA_ITEM,INVEST_CHANNEL,QUOTA_CONTAIN_ID,QUOTA_NUM,UNIT) values('申购最低额','直销中心柜台','含',100000,'元'),('追加申购最低额','直销中心柜台','含',100000,'元');

"""

start_insert="\n 请只输出sql，其他的不要输出。 \n 请只输出sql，而且输出以insert开始。"

def json2sql(jsondata):
    prompt = prompt_prefix + jsondata, start_insert
    completion = openai.Completion.create(
            engine="code-davinci-002", 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=1500, 
            n=1, 
            stop=[";"])
    return completion.choices[0].text
