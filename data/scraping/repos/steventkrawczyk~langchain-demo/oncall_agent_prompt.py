from langchain.prompts import PromptTemplate

prompt_template = """You are a Site Reliability Engineer at Amazon.
Use the runbook to find the correct command to run to address the error.
Then, write that command.
    Runbook: {runbook}
    Error: {error}
    Command:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["runbook", "error"]
)