from langchain.prompts import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
few_shot_human4 = HumanMessagePromptTemplate.from_template("1、投资人申购、赎回的基金份额需为最小申购、赎回单位的整数倍。最小申购、赎回单位为1万份。")


result1='''
{{
	"限额项目": "最小申购赎回单位",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1万",
	"单位": "份"
}}
'''

few_shot_ai_template4 = f'''
[
	{result1}
]
'''

few_shot_ai4 = AIMessagePromptTemplate.from_template(few_shot_ai_template4)