
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from sql_agents.base_agent import BaseAgent
from utils.timer import Timer
import logging

ZERO_SHOT_PROMPT = """
Database schema in the form of CREATE_TABLE statements:

{database_schema}

Using valid SQL, answer the following question based on the tables provided above. 
It is important to use qualified column names in the SQL-query, meaning the form 
\"SELECT table_name.column_name FROM table_name;\"

Hint helps you to write the correct sqlite SQL query.
Question: {question}
Hint: {evidence}
DO NOT return anything else except the SQL query.
"""

class ZeroShotAgent(BaseAgent):
    total_tokens = 0
    prompt_tokens = 0 
    total_cost = 0
    completion_tokens = 0
    last_call_execution_time = 0
    total_call_execution_time = 0

    def __init__(self, llm):        
        self.llm = llm

        self.prompt_template = ZERO_SHOT_PROMPT
        prompt = PromptTemplate(    
            input_variables=["question", "database_schema","evidence"],
            template=ZERO_SHOT_PROMPT,
        )

        self.chain = LLMChain(llm=llm, prompt=prompt)

    def generate_query(self, database_schema, question, evidence):
        with get_openai_callback() as cb:
            with Timer() as t:
                response = self.chain.run({
                    'database_schema': database_schema,
                    'question': question,
                    "evidence": evidence
                })

            logging.info(f"OpenAI API execution time: {t.elapsed_time:.2f}")
            
            self.last_call_execution_time = t.elapsed_time
            self.total_call_execution_time += t.elapsed_time
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.total_cost += cb.total_cost
            self.completion_tokens += cb.completion_tokens

            return response
        
    