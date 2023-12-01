from typing import Dict, List
import openai
from langchain.llms import OpenAI
from langchain import SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain

from assistant.approach import Approach
from config.config import logger
from config.prompt import sql_generator, sql_agent


class SQLAssistant(Approach):
    def __init__(
            self,  
            gpt_deployment: str
            ):
        self.gpt_deployment = gpt_deployment
        self.prompt = sql_agent[1]

    def run(self, history: List, db) -> any:
        natural_query = history[-1]["content"]
        llm = OpenAI(
            model=self.gpt_deployment, 
            temperature=0.3,
            openai_api_key=openai.api_key
        )

        # db_chain = SQLDatabaseChain.from_llm(
        #     llm, 
        #     db, 
        #     verbose=True, 
        #     return_intermediate_steps=True,
        #     top_k=10)
        
        db_chain = SQLDatabaseSequentialChain.from_llm(
            llm=llm, 
            database=db,
            # return_intermediate_steps=True,
            verbose=True, 
            top_k=10,
            use_query_checker=True
            )
        
        # response = db_chain({"query": natural_query})
        # response = db_chain.run(self.prompt.format(question=natural_query))
        response = db_chain.run(natural_query)

        # output_query = response["intermediate_steps"][0]["input"]
        # output_query = output_query[output_query.find("SQLQuery:"):-8]
        # return {"data_points": output_query, "answer": response["result"], "thoughts": output_query}
        return {"data_points": "", "answer": response, "thoughts": ""}

class SQLGenerator(Approach):
    def __init__(
            self,  
            gpt_deployment: str
            ):
        self.gpt_deployment = gpt_deployment
    
    def run(self, conversation: Dict) -> Dict:

        natural_query = conversation[-1]["content"]
        response = openai.ChatCompletion.create(
            model=self.gpt_deployment,
            messages=[
                {
                "role": "system",
                "content": sql_generator["system"]
                },
                {
                "role": "user",
                "content": sql_generator["user"].format(input_user=natural_query)
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        
        return {"data_points": "...", "answer": response["choices"][0]['message']['content'], "thoughts": "..."}