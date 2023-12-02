"""
The `DelegatorChain` takes the task given by the user, and dynamically decides, depending on the 
complexity of the problem whether it should be a **two-way** or **groupchat** interactions,
what agents should be assigned to the job and what datasource should be used. 
Context of the entire ecosystem is fed in-context to inform the decision process. 

Written by Aaron Ward - November 12th 2023
"""
import os
import yaml
import logging
from pathlib import Path
import langchain
import autogen
from langchain.output_parsers.list import CommaSeparatedListOutputParser

logger = logging.getLogger(__name__)

# TODO: get available functions
# TODO: Get available datasources
# TODO: Add "Can't perform this action" and early break if not possible
# TODO: Implement dynamic teams assignment

class DelegatorChain():
    def __init__(self, openai_api_key):
        super().__init__()
        self.openai_api_key = openai_api_key
        self.llm_chains = []  # Buffer to store LLMChain objects

        # Add LLMChain objects to the buffer
        # self.add_chain(self.get_plan_chain())
        # self.add_chain(self.get_functions())
        # self.add_chain(self.get_team())

    def add_chain(self, llm_chain):
        self.llm_chains.append(llm_chain)
    
    def get_roles(self):
        system_messages_path = "chains/configs/roles.yaml"
    
        try:
            with open(system_messages_path, 'r') as file:
                roles_config = yaml.safe_load(file)
            return roles_config
        except Exception as e:
            logger.error(f"Failed to read roles configuration: {e}")
            raise
    
    def get_config_list(self):
        """
        Get a list of configuration options for Autogen.

        Returns:
            list: List of configuration options.
        """
        config_list = autogen.config_list_from_dotenv(
            dotenv_file_path=self.env_path,
            model_api_key_map={
                "gpt-4": "OPENAI_API_KEY",
                "gpt-3.5-turbo": "OPENAI_API_KEY",
            },
            filter_dict={
                "model": "gpt-4"
            }
        )

        if not config_list:
            raise ValueError("Config list wasn't set, check configurations.")
        logging.info("Models to use: %s", [c["model"] for c in config_list])
        return config_list

    def get_template(self, prompt, input):
        # Create the first PromptTemplate
        prompt = langchain.PromptTemplate(
            input_variables=input,
            template=prompt.PROMPT_STRING
        )
        return prompt

    # def get_plan_chain(self):
    #     prompt_1 = self.get_template(planner, input=["input"])

    #     return langchain.LLMChain(llm=langchain.OpenAI(
    #         openai_api_key=self.openai_api_key), prompt=prompt_1)

    # def get_list_chain(self):
    #     prompt_2 = self.get_template(list_parsing, input=["input"])
    #     output_parser = CommaSeparatedListOutputParser()

    #     return langchain.LLMChain(llm=langchain.OpenAI(
    #         openai_api_key=self.openai_api_key), prompt=prompt_2, output_parser=output_parser)

    # def get_breakdown_chain(self):
    #     prompt_3 = self.get_template(task_breakdown, input=["input", "task", "previous_task"])

    #     return langchain.LLMChain(llm=langchain.OpenAI(
    #         openai_api_key=self.openai_api_key), prompt=prompt_3)
    
    # def get_repo_admin_chain(self):
    #     prompt_4 = self.get_template(repo_admin, input=["input", "tasks"])
    #     output_parser = CommaSeparatedListOutputParser()

    #     return langchain.LLMChain(llm=langchain.OpenAI(
    #         openai_api_key=self.openai_api_key), prompt=prompt_4, output_parser=output_parser)

    def run(self, input_text):
        logger.info('Running Delegator chain...')
        llm_1 = self.llm_chains[0].run(input_text)
        llm_2 = self.llm_chains[1].run(llm_1)
        llm_3 = self.llm_chains[2].run(llm_2)

        return {
            "Planner": llm_1,
            "List Parsing Chain": llm_2,
        }
    


##################



# def get_config_list(self):
#     """
#     Get a list of configuration options for Autogen.

#     Returns:
#         list: List of configuration options.
#     """
#     config_list = autogen.config_list_from_dotenv(
#         dotenv_file_path=self.env_path,
#         model_api_key_map={
#             "gpt-4": "OPENAI_API_KEY",
#             "gpt-3.5-turbo": "OPENAI_API_KEY",
#         },
#         filter_dict={
#             "model": {self.model}
#         }
#     )

#     if not config_list:
#         raise ValueError("Config list wasn't set, check configurations.")
#     logging.info("Models to use: %s", [c["model"] for c in config_list])
#     return config_list


# def get_config_common(self):
#     """ Get the common confirugations among all agents"""
#     return {
#         "request_timeout": 1000,
#         "seed": 42,
#         # "use_cache": False, # 
#         "config_list": self.config_list,
#         "temperature": self.temperature,
#         "functions": [
#             {
#             "name": "run_sql",
#             "description": "Run a SQL query against the postgres database",
#             "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "sql": {
#                             "type": "string",
#                             "description": "The SQL query to run"
#                         }
#                     },
#                     "required": ["sql"]
#                 }
#             }
#         ]
#     }


# def get_system_messages(self):
#     """
#     Get system messages for different agent roles.
#     TODO: Move this out to it's own location as the application grows
#     Returns:
#         dict: A dictionary of system messages.
#     """

#     # This is supposed to pull in 

#     system_messages = {
#         "Senior_Data_Analyst": 
#             "Senior Data Analyst. You follow an approval plan. You run the SQL query. Generate the final response and send it to the Product Manager for review.",
#         "Admin": 
#             "A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin.",
#         "Data_Engineer": 
#             "Data Engineer. You follow an approval plan. Think about and generate SQL based on the requirements provided. Send it out for review.",
#         "Senior_Data_Analyst": 
#             "Senior Data Analyst. You follow an approval plan. You run the SQL query. Generate the final response and send it to the Product Manager for review.",
#         "Product_Manager": 
#             "Product Manager. Validate the response to ensure it's correct.",
#     }

#     return system_messages

# def get_function_map(self):
#     """""
#     Get a mapping of functions to their database operation counterparts.

#     Returns:
#         dict: A dictionary mapping function names to database methods.
#     """
#     function_map = {    
#         "run_sql": db.run_sql,
#     }
#     return function_map


# ######
# # Here is example full config
# user_proxy_config = {
#     "name": "User_Proxy",
#     "code_execution_config": {
#             "work_dir": "some_dir/",
#             "use_docker": False
#     },
#     "human_input_mode": "NEVER",
#     # "is_termination_msg": self.is_termination_message,
#     "function_map": get_function_map(),
#     "llm_config": {
#         "request_timeout": 1000,
#         "seed": 42,
#         # "use_cache": False, # 
#         "config_list": get_config_list(),
#         "temperature": 0.2,
#         "functions": [
#             {
#             "name": "run_sql",
#             "description": "Run a SQL query against the postgres database",
#             "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "sql": {
#                             "type": "string",
#                             "description": "The SQL query to run"
#                         }
#                     },
#                     "required": ["sql"]
#                 }
#             }
#         ]
#     },
# }




# def instantiate_two_way_chat(self):
#     logging.info("Initializing Agents")

#     llm_config_common = self.get_config_common()
#     system_messages = self.get_system_messages()

#     user_proxy_config = {
#         "name": "User_Proxy",
#         "code_execution_config": {
#                 "work_dir": self.work_dir,
#                 "use_docker": False
#             },
#         "human_input_mode": "NEVER",
#         # "is_termination_msg": self.is_termination_message,
#         "llm_config": llm_config_common,
#         "function_map": self.get_function_map()
#     }

#     data_engineer_config = {
#         "name": "Data_Engineer",
#         "llm_config": llm_config_common,
#         "system_message": system_messages["Data_Engineer"],
#         "code_execution_config": False,
#         "human_input_mode": "NEVER",
#         # "is_termination_msg": self.is_termination_message,
#     }

#     user_proxy = UserProxyAgent(**user_proxy_config)
#     data_engineer = AssistantAgent(**data_engineer_config)

#     self.two_way_user_proxy = user_proxy
#     self.two_way_secondary_agent = data_engineer

# def instantiate_groupchat(self):
#     # TODO: Ramblings of a mad man, but imagine
#     # you could pass this function to a conversation
#     # and allow the agents to spin off their own sub agent conversations
#     # if they need to create a spin off task using different agents.
#     logging.info("Initializing Agents")

#     llm_config_common = self.get_config_common()
#     system_messages = self.get_system_messages()

#     user_proxy_config = {
#         "name": "Admin",
#         "system_message": system_messages["Admin"],
#         "code_execution_config": False,
#         "human_input_mode": "NEVER",
#         # "is_termination_msg": self.is_termination_message,
#     }
#     data_engineer_config = {
#         "name": "Data_Engineer",
#         "llm_config": llm_config_common,
#         "system_message": system_messages["Data_Engineer"],
#         "code_execution_config": False,
#         "human_input_mode": "NEVER",
#         # "is_termination_msg": self.is_termination_message,
#     }
#     senior_data_analyst_config = {
#         "name": "Senior_Data_Analyst",
#         "system_message": system_messages["Senior_Data_Analyst"],
#         "human_input_mode": "NEVER",
#         "code_execution_config": {
#             "work_dir": self.work_dir
#         },
#         # "is_termination_msg": self.is_termination_message,
#         "function_map": self.get_function_map()
#     }
#     product_manager = {
#         "name": "Product_Manager",
#         "llm_config": llm_config_common,
#         "system_message": system_messages["Product_Manager"],
#         "code_execution_config": False,
#         "human_input_mode": "NEVER",
#         # "is_termination_msg": self.is_termination_message,
#     }

#     # For agents in UserProxyAgents:
#     #   create agents with corresponding config
#     # for agents in AssistantAgents:
#     #   create agents

#     # Instantiate agents using the above configurations
#     user_proxy = autogen.UserProxyAgent(**user_proxy_config)
#     data_engineer = autogen.AssistantAgent(**data_engineer_config)
#     senior_data_analyst = autogen.AssistantAgent(**senior_data_analyst_config)
#     product_manager = autogen.AssistantAgent(**product_manager)

#     # Instantiate group chat and manager
#     groupchat_user_proxy = autogen.GroupChat(agents=[user_proxy, data_engineer, product_manager, senior_data_analyst], messages=[], max_round=50)
#     groupchat_manager = autogen.GroupChatManager(groupchat=groupchat_user_proxy, llm_config=llm_config_common)

#     self.groupchat_user_proxy = groupchat_user_proxy
#     self.groupchat_secondary_agent = groupchat_manager
#     print("Agents Initiated!")