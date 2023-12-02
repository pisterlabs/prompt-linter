# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-04-29
# 不积跬步 无以至千里

import utils
import prompt
import agent
import memory

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class KuiBu():
    def __init__(self, name, personalities):
        # LLM settings
        self.smart_llm = ChatOpenAI(model_name = "gpt-4", temperature=0)
        self.fast_llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)
        self.smart_llm_token_limit = 8000
        self.fast_llm_token_limit = 4000
        self.recursion_depth = 3
        self.agent = agent.Agent(name, personalities)

    def gate(self, query):
        # Check the difficulty of the given query
        print("Check the difficulty of the given query")

        # Retrieve short-term memory
        short_term_memory_documents = self.agent.short_term_memory.query(query, top_k=5)
        short_term_memory_messages = self.agent.short_term_memory.convert(short_term_memory_documents)

        messages = prompt.gate_prompt.format_messages(query=query)
        result = self.fast_llm(short_term_memory_messages + messages)
        print(f"Gate Result: {result}")
        return result.content != "NO"
    
    def kui(self, query_object):
        # Generate plan for the give query
        query = query_object["step_description"]

        # Get kuibu template
        kuibu_template = prompt.kuibu_template.replace("{{question}}", query)
        kuibu_message = HumanMessage(content=kuibu_template)

        # Retrieve short-term memory
        short_term_memory_documents = self.agent.short_term_memory.query(query, top_k=3)
        short_term_memory_messages = self.agent.short_term_memory.convert(short_term_memory_documents)

        # Retrieve history
        history = self.agent.history.query(top_k=5)

        # Inference
        print(f"Planing...")
        result = self.smart_llm(short_term_memory_messages + history + [kuibu_message])
        
        # Validation
        step_list = utils.kuibu_validation(result.content)
        step_list = step_list if step_list else [query_object]

        return step_list

    def bu(self, step_list, recursion_level=0, socket_config=None):
        # Iterative execution
        step_result = list()

        for step in step_list:
            utils.send_message(f"🚧 Check if it is a hard query: {step['step_description']}", "system", socket_config)
            gate_pass = self.gate(step["step_description"])
            utils.send_message(f"🚧 Kuibu Check: {gate_pass}", "system", socket_config)

            if gate_pass and recursion_level <= self.recursion_depth:
                utils.send_message(f"🚧 Kuibu Planning...", "system", socket_config)
                plan_list = self.kui(step)
                
                for index, plan in enumerate(plan_list):
                    utils.send_message(f"Step_{index} {plan['step_description']} -> {plan['step_result_needed']}", "system", socket_config)
                
                result = self.bu(plan_list, recursion_level + 1, socket_config)

            else:
                if socket_config:
                    utils.send_message(f"[KuiBu] Looking for: {step['step_description']}", "system", socket_config)
                    result = self.agent.receive(f"{step['step_description']}", socket_config)
                else:
                    result = self.agent.execute(f"{step['step_description']}")
            
            step_result += [{
                "step_description": step["step_description"],
                "step_result": result
            }]

        return result

if __name__ == "__main__":
    kuibu = KuiBu("CISCO_BOT", ["help customers solving their problems"])
    kuibu.bu([
        {
            "step_description": "The 1995 Tooheys 1000 driver who has second-to-last in the Tooheys Top 10 was born where?",
            "step_result_needed": "the birthplace of the driver",
        }
    ])
