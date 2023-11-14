from langchain.prompts import (
    SystemMessagePromptTemplate
)


systemTemplate='''You are an assistant designed to extract entities from text. Users will paste in a string of text and you will respond with entities you've extracted from the text as a JSON object.
Here's your output format:
{sample}
'''
sample ='''
{
  "限额项目": "",
  "销售方式": "",
  "是否含申购费": "",
  "金额数": "",
  "单位": ""
}
'''

system_message_prompt = SystemMessagePromptTemplate.from_template(systemTemplate,sample=sample)