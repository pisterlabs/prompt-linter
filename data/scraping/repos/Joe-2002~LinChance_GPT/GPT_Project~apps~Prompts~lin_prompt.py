from langchain import PromptTemplate

company_template = """
你是一名精通公司业务的培训人员

给我返回一个相关业务的要点和注意事项，要求语言简洁明了. 它应该和上述提供的文本内容有关.

关于{company_desription} 这个业务，有什么需要注意的呢?
"""

#创建一个prompt模板
prompt_template=PromptTemplate(
               input_variables=["company_desription"], 
               template=company_template
               )