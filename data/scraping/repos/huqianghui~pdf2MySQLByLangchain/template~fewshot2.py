from langchain.prompts import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

few_shot_human_template2 = "通过基金销售机构及公司直销中心(含网上直销系统)首次申购或追加申购各类基金份额时,单笔最低金额为人民币0.01元(含申购费)"
few_shot_human2 = HumanMessagePromptTemplate.from_template(few_shot_human_template2)

result1='''
{{
	"限额项目": "申购最低额",
	"销售方式": "销售机构/直销中心/网上直销系统",
	"是否含申购费": "含",
	"金额数": "0.01",
	"单位": "元"
}}
'''

result2='''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "销售机构/直销中心/网上直销系统",
	"是否含申购费": "含",
	"金额数": "0.01",
	"单位": "元"
}}
'''

few_shot_ai_template2 = f'''
[
	{result1},
	{result2}
]

'''


few_shot_ai2 = AIMessagePromptTemplate.from_template(few_shot_ai_template2)