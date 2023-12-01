from repolya._const import AUTOGEN_CONFIG
from repolya._log import logger_autogen

from repolya.autogen.db_postgre import (
    PostgresManager,
    PostgresAgentInstruments,
)
from repolya.autogen.tool_function import (
    _def_run_postgre,
    _def_write_file,
    _def_write_json_file,
    _def_write_yaml_file,
)
from repolya.autogen.organizer import (
    Organizer,
)

from autogen import(
    ConversableAgent,
    UserProxyAgent,
    AssistantAgent,
    config_list_from_json,
    Agent,
)

from typing import Optional, List, Dict, Any
import guidance


config_list = config_list_from_json(env_or_file=str(AUTOGEN_CONFIG))


# Base Configuration
base_config = {
    "config_list": config_list,
    "timeout": 120,
    "temperature": 0,
    "model": "gpt-4",
    # "use_cache": False,
    # "seed": 42,
}


# def is_termination_msg(content):
#     have_content = content.get("content", None) is not None
#     if have_content and "APPROVED" in content["content"]:
#         return True
#     return False

# COMPLETION_PROMPT = "If everything looks good, respond with 'APPROVED'."


# # takes in the prompt and manages the group chat
# USER_PROMPT = (
#     "A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin."
#     # "A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin. " + COMPLETION_PROMPT
# )
# POSTGRE_user = UserProxyAgent(
#     name="POSTGRE_user",
#     code_execution_config=False,
#     human_input_mode="NEVER",
#     is_termination_msg=is_termination_msg,
#     system_message=USER_PROMPT,
# )


# # generates the sql query
# ENGINEER_PROMPT = (
#     "A Data Engineer. Generate the initial SQL based on the requirements provided. Send it to the Sr Data Analyst to be executed."
#     # "A Data Engineer. You follow an approved plan. Generate the initial SQL based on the requirements provided. Send it to the Sr Data Analyst for review. " + COMPLETION_PROMPT
# )
# POSTGRE_engineer = AssistantAgent(
#     name="POSTGRE_engineer",
#     code_execution_config=False,
#     human_input_mode="NEVER",
#     is_termination_msg=is_termination_msg,
#     llm_config=base_config,
#     system_message=ENGINEER_PROMPT,
# )


# # run the sql query and generate the response
# ANALYST_PROMPT = (
#     "Sr Data Analyst. You run the SQL query using the run_postgre function, send the raw response to the data viz team. You use the run_postgre function" + " to generate the response and send it to the product manager for final review."
#     # "Sr Data Analyst. You follow an approved plan. You run the SQL query, generate the response and send it to the product manager for final review. " + COMPLETION_PROMPT
# )
# # POSTGRE_analyst = AssistantAgent(
# #     name="POSTGRE_analyst",
# #     code_execution_config=False,
# #     human_input_mode="NEVER",
# #     is_termination_msg=is_termination_msg,
# #     llm_config={
# #         **base_config,
# #         "functions": [_def_run_postgre],
# #     },
# #     function_map={
# #         'run_postgre': run_postgre,
# #     },
# #     system_message=ANALYST_PROMPT,
# # )
# def build_sr_data_analyst_agent(db: PostgresManager):
#     return AssistantAgent(
#         name="POSTGRE_analyst",
#         code_execution_config=False,
#         human_input_mode="NEVER",
#         is_termination_msg=is_termination_msg,
#         llm_config={
#             **base_config,
#             "functions": [_def_run_postgre],
#         },
#         function_map={
#             'run_postgre': db.run_postgre,
#         },
#         system_message=ANALYST_PROMPT,
#     )


# # validate the response to make sure it's correct
# PM_PROMPT = (
#     "Product Manager. Validate the response to make sure it's correct. " + COMPLETION_PROMPT
# )
# POSTGRE_pm = AssistantAgent(
#     name="POSTGRE_manager",
#     code_execution_config=False,
#     human_input_mode="NEVER",
#     is_termination_msg=is_termination_msg,
#     llm_config=base_config,
#     system_message=PM_PROMPT,
# )


# def build_data_eng_team(agent_instruments: PostgresAgentInstruments):
#     # create a set of agents with specific roles
#     # admin user proxy agent - takes in the prompt and manages the group chat
#     POSTGRE_user = UserProxyAgent(
#         name="Admin",
#         code_execution_config=False,
#         human_input_mode="NEVER",
#         system_message=USER_PROMPT,
#     )

#     # data engineer agent - generates the sql query
#     POSTGRE_engineer = AssistantAgent(
#         name="Engineer",
#         code_execution_config=False,
#         human_input_mode="NEVER",
#         llm_config=base_config,
#         system_message=ENGINEER_PROMPT,
#     )

#     POSTGRE_analyst = AssistantAgent(
#         name="Sr_Data_Analyst",
#         code_execution_config=False,
#         human_input_mode="NEVER",
#         llm_config={
#             **base_config,
#             "functions": [_def_run_postgre],
#         },
#         function_map={
#             'run_postgre': agent_instruments.run_postgre,
#         },
#         system_message=ANALYST_PROMPT,
#     )

#     # product manager – validate the response to make sure it's correct
#     POSTGRE_pm = AssistantAgent(
#         name="Product_Manager",
#         code_execution_config=False,
#         human_input_mode="NEVER",
#         llm_config=base_config,
#         system_message=PM_PROMPT,
#     )


USER_PROXY_PROMPT = "A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin."
DATA_ENGINEER_PROMPT = "A Data Engineer. Generate the initial SQL based on the requirements provided. Send it to the Sr Data Analyst to be executed. "
SR_DATA_ANALYST_PROMPT = "Sr Data Analyst. You run the SQL query using the run_postgre function, send the raw response to the data viz team. You use the run_postgre function exclusively."

GUIDANCE_SCRUM_MASTER_SQL_NLQ_PROMPT = """
Is the following block of text a SQL Natural Language Query (NLQ)? Please rank from 1 to 5, where:
1: Definitely not NLQ
2: Likely not NLQ
3: Neutral / Unsure
4: Likely NLQ
5: Definitely NLQ

Return the rank as a number exclusively using the rank variable to be casted as an integer.

Block of Text: {{potential_nlq}}
{{#select "rank" logprobs='logprobs'}} 1{{or}} 2{{or}} 3{{or}} 4{{or}} 5{{/select}}
"""

DATA_INSIGHTS_GUIDANCE_PROMPT = """
You're a data innovator. You analyze SQL databases table structure and generate 3 novel insights for your team to reflect on and query. 
Format your insights in JSON format.
```json
[{{#geneach 'insight' num_iterations=3 join=','}}
{
    "insight": "{{gen 'insight' temperature=0.7}}",
    "actionable_business_value": "{{gen 'actionable_value' temperature=0.7}}",
    "sql": "{{gen 'new_query' temperature=0.7}}"
}
{{/geneach}}]
```"""

INSIGHTS_FILE_REPORTER_PROMPT = "You're a data reporter. You write json data you receive directly into a file using the write_innovation_file function."

TEXT_REPORT_ANALYST_PROMPT = "Text File Report Analyst. You exclusively use the write_file function on a summarized report."
JSON_REPORT_ANALYST_PROMPT = "Json Report Analyst. You exclusively use the write_json_file function on the report."
YML_REPORT_ANALYST_PROMPT = "Yaml Report Analyst. You exclusively use the write_yml_file function on the report."

# # unused prompts
# COMPLETION_PROMPT = "If everything looks good, respond with APPROVED"
# PRODUCT_MANAGER_PROMPT = (
#     "Product Manager. Validate the response to make sure it's correct"
#     + COMPLETION_PROMPT
# )


# Configuration with "run_postgre"
run_postgre_config = {
    **base_config,  # Inherit base configuration
    "functions": [_def_run_postgre],
}

# Configuration with "write_file"
write_file_config = {
    **base_config,  # Inherit base configuration
    "functions": [_def_write_file],
}

# Configuration with "write_json_file"
write_json_file_config = {
    **base_config,  # Inherit base configuration
    "functions": [_def_write_json_file],
}

write_yaml_file_config = {
    **base_config,  # Inherit base configuration
    "functions": [_def_write_yaml_file],
}


write_innovation_file_config = {
    **base_config,  # Inherit base configuration
    "functions": [
        {
            "name": "write_innovation_file",
            "description": "Write a file to the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content of the file to write",
                    },
                },
                "required": ["content"],
            },
        }
    ],
}


def build_data_eng_team(agent_instruments: PostgresAgentInstruments):
    """
    Build a team of agents that can generate, execute, and report an SQL query
    """
    # create a set of agents with specific roles
    # admin user proxy agent - takes in the prompt and manages the group chat
    user_proxy = UserProxyAgent(
        name="Admin",
        system_message=USER_PROXY_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    # data engineer agent - generates the sql query
    data_engineer = AssistantAgent(
        name="Engineer",
        llm_config=base_config,
        system_message=DATA_ENGINEER_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    sr_data_analyst = AssistantAgent(
        name="Sr_Data_Analyst",
        llm_config=run_postgre_config,
        system_message=SR_DATA_ANALYST_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
        function_map={
            "run_postgre": agent_instruments.run_postgre,
        },
    )
    return [
        user_proxy,
        data_engineer,
        sr_data_analyst,
    ]


def build_data_viz_team(agent_instruments: PostgresAgentInstruments):
    # admin user proxy agent - takes in the prompt and manages the group chat
    user_proxy = UserProxyAgent(
        name="Admin",
        system_message=USER_PROXY_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    # text report analyst - writes a summary report of the results and saves them to a local text file
    text_report_analyst = AssistantAgent(
        name="Text_Report_Analyst",
        llm_config=write_file_config,
        system_message=TEXT_REPORT_ANALYST_PROMPT,
        human_input_mode="NEVER",
        function_map={
            "write_file": agent_instruments.write_file,
        },
    )
    # json report analyst - writes a summary report of the results and saves them to a local json file
    json_report_analyst = AssistantAgent(
        name="Json_Report_Analyst",
        llm_config=write_json_file_config,
        system_message=JSON_REPORT_ANALYST_PROMPT,
        human_input_mode="NEVER",
        function_map={
            "write_json_file": agent_instruments.write_json_file,
        },
    )
    yaml_report_analyst = AssistantAgent(
        name="Yml_Report_Analyst",
        llm_config=write_yaml_file_config,
        system_message=YML_REPORT_ANALYST_PROMPT,
        human_input_mode="NEVER",
        function_map={
            "write_yml_file": agent_instruments.write_yml_file,
        },
    )
    return [
        user_proxy,
        text_report_analyst,
        json_report_analyst,
        yaml_report_analyst,
    ]


def build_scrum_master_team(agent_instruments: PostgresAgentInstruments):
    user_proxy = UserProxyAgent(
        name="Admin",
        system_message=USER_PROXY_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    scrum_agent = DefensiveScrumMasterAgent(
        name="Scrum_Master",
        llm_config=base_config,
        system_message=GUIDANCE_SCRUM_MASTER_SQL_NLQ_PROMPT,
        human_input_mode="NEVER",
    )
    return [
        user_proxy,
        scrum_agent,
    ]


def build_insights_team(agent_instruments: PostgresAgentInstruments):
    user_proxy = UserProxyAgent(
        name="Admin",
        system_message=USER_PROXY_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    insights_agent = InsightsAgent(
        name="Insights",
        llm_config=base_config,
        system_message=DATA_INSIGHTS_GUIDANCE_PROMPT,
        human_input_mode="NEVER",
    )
    insights_data_reporter = AssistantAgent(
        name="Insights_Data_Reporter",
        llm_config=write_innovation_file_config,
        system_message=INSIGHTS_FILE_REPORTER_PROMPT,
        human_input_mode="NEVER",
        function_map={
            "write_innovation_file": agent_instruments.write_innovation_file,
        },
    )
    return [
        user_proxy,
        insights_agent,
        insights_data_reporter,
    ]


def build_team_organizer(
    team: str,
    agent_instruments: PostgresAgentInstruments,
    validate_results: callable = None,
) -> Organizer:
    """
    Based on a team name, build a team of agents and return an orchestrator
    """
    if team == "data_eng":
        return Organizer(
            name="data_eng_team",
            agents=build_data_eng_team(agent_instruments),
            agent_instruments=agent_instruments,
            validate_results_func=validate_results,
        )
    elif team == "data_viz":
        return Organizer(
            name="data_viz_team",
            agents=build_data_viz_team(agent_instruments),
            validate_results_func=validate_results,
        )
    elif team == "scrum_master":
        return Organizer(
            name="scrum_master_team",
            agents=build_scrum_master_team(agent_instruments),
            agent_instruments=agent_instruments,
            validate_results_func=validate_results,
        )
    elif team == "data_insights":
        return Organizer(
            name="data_insights_team",
            agents=build_insights_team(agent_instruments),
            agent_instruments=agent_instruments,
            validate_results_func=validate_results,
        )
    raise Exception("Unknown team: " + team)


class DefensiveScrumMasterAgent(ConversableAgent):
    """
    Custom agent that uses the guidance function to determine if a message is a SQL NLQ
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the new reply function for this specific agent
        self.register_reply(self, self.check_sql_nlq, position=0)

    def check_sql_nlq(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,  # Persistent state.
    ):
        # Check the last received message
        last_message = messages[-1]["content"]
        # Use the guidance string to determine if the message is a SQL NLQ
        response = guidance(
            GUIDANCE_SCRUM_MASTER_SQL_NLQ_PROMPT, potential_nlq=last_message
        )
        # You can return the exact response or just a simplified version,
        # here we are just returning the rank for simplicity
        rank = response.get("choices", [{}])[0].get("rank", "3")
        return True, rank


class InsightsAgent(ConversableAgent):
    """
    Custom agent that uses the guidance function to generate insights in JSON format
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_reply(self, self.generate_insights, position=0)

    def generate_insights(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ):
        insights = guidance(DATA_INSIGHTS_GUIDANCE_PROMPT)
        return True, insights

