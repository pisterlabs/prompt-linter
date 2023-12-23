from langchain.prompts import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

few_shot_human5 = HumanMessagePromptTemplate.from_template("投资人可将其全部或部分基金份额赎回。每类基金份额单笔赎回或转换不得少于1份\n(如该账户在该销售机构托管的该类基金份额余额不足1份,则必须一次性赎回或转出该类\n基金份额全部份额);若某笔赎回将导致投资人在该销售机构托管的该类基金份额余额不足1份时,基金管理人有权将投资人在该销售机构托管的该类基金份额剩余份额一次性全部赎回。")

result1='''
{{
	"限额项目": "赎回最低额",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
'''

result2='''
{{
	"限额项目": "转换最低额",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
'''

result3='''
{{
	"限额项目": "账户持有份额下限",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
'''

few_shot_ai_template5 = f'''
[
	{result1},
	{result2},
	{result3}
]
'''

few_shot_ai5 = AIMessagePromptTemplate.from_template(few_shot_ai_template5)
