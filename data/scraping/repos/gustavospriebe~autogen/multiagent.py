import openai
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from decouple import config

openai.api_key = config('OPENAI_API_KEY')

config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo"]})


planner = AssistantAgent("planner", llm_config={
    "config_list": config_list},
    # the default system message of the AssistantAgent is overwritten here
    system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning. convert it to a step which can be implemented by writing code. For example, the action of browsing the web can be implemented by writing code which reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix."
)


planner_user = UserProxyAgent(
    "planner_user", code_execution_config={"work_dir": "coding"}
    # , max_consecutive_auto_reply=0, # terminate without auto_reply
    # human_input_mode="NEVER"
)


def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    return planner_user.last_message()['content']


assistant = AssistantAgent('assistant', llm_config={
    "temperature": 0,
    "request_timeout": 600,
    "seed": 42,
    "model": "gpt-3.5-turbo",
    "config_list": autogen.config_list_openai_aoai(exclude='aoai'),
    "functions": [{
        "name": "ask_planner",
        "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest a new plan.",
        "parameters": {
            "message": {
                "type": "string",
                "description": "question to ask planner. Make sure the question inclide enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner."
            }
        },
        "required": ["message"]
    }]
})


user_proxy = UserProxyAgent('user_proxy',
                            # human_input_mode="TERMINATE",
                            max_consecutive_auto_reply=5,
                            code_execution_config={
                                "work_dir": "planning", "use_docker": True,
                            }, function_map={"ask_planner": ask_planner})



user_proxy.initiate_chat(assistant, message="build a simple snake game. use subprocess for pip installs")
