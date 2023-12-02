from langchain.prompts import PromptTemplate

prompt_template = """You are a Site Reliability Manager at Amazon.
Given a runbook, error, and the command run by your Site Reliability Engineer, grade them based on whether or not they took the correct action.
Give them a grade of SUCCESS or FAILURE and explain why you gave that grade.
    Runbook: {runbook}
    Error: {error}
    Command: {command}
    Grade:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["runbook", "error", "command"]
)