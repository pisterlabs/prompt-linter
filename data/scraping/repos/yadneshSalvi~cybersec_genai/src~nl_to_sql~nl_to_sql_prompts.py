from langchain.prompts import PromptTemplate

system_prompt_general = PromptTemplate(
        input_variables=[],
        template="""
You are a helpful assistant.
"""
    )

system_prompt_sql = PromptTemplate(
        input_variables=[],
        template="""
You are an expert in writing SQL queries.
""",
    )

prompt_get_most_relevant_severity = PromptTemplate(
        input_variables=["user_query","severities"],
        template="""
Your task is to return the most relevant severity from the given list of severities for the given question.

"Question:{user_query}"

"Severities :\n {severities}"

Output should be in following json format only:
{{"most relevant severity":""}}

"""
)

tables_info = """
Table 1:
asset = {
company_id,
device_id,
user_id,
device_type,
device_subtype
ip_info,
os_info,
bios_info,
ip_info
}
Table 2:
misconfig = {
company_id,
device_id,
vulnerability,
severity,
threat_level
}
"""

prompt_get_sql_query = PromptTemplate(
        input_variables=["tables_info", "severity_value"],
        template="""
Your task is to construct an SQL where severity is {severity_value} over the SQL tables containing following columns. You may have to join several tables.

SQL tables are as below:\n {tables_info}

Strictly return only the following json in output with correct json format:
{{"sql_query":""}}
"""
)