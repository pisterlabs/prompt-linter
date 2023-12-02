from langchain.prompts.prompt import PromptTemplate

default_template = """You are a sales development representative for a small AI startup.
The startup helps software development teams evaluate LLMs for use in their products.
Draft a cold outreach email for a potential customer, based on the JSON input provided below:
----------------
{text}"""
SDR_PROMPT = PromptTemplate.from_template(default_template)