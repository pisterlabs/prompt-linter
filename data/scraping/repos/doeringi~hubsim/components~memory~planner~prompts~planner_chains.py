from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from components.llm import LocalLLM

llm = LocalLLM()

plan_day_template = """
You are {agent_first_name} {agent_last_name}. 
Your biography: {agent_bio}. 
Yesterday you did the following: {yesterday_summary}.
Today is {datetime_today}.
You are planning your day in broad-strokes (five to eight items):
1) wake up at {datetime_wakeup} and finish morning routine 
2) 
"""

plan_day_prompt_template = PromptTemplate(
    input_variables=[
        "agent_first_name",
        "agent_last_name",
        "agent_bio",
        "yesterday_summary",
        "datetime_today",
        "datetime_wakup",
    ],
    template=plan_day_template,
)

plan_day_chain = LLMChain(llm=llm.get_llm, prompt_template=plan_day_prompt_template)
