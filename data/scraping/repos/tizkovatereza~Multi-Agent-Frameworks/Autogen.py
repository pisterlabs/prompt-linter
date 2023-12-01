# INSTALL AUTOGEN IN TERMINAL

# pip install pyautogen
# pip install "pyautogen[blendsearch]" for optional dependencies


# IMPORT PACKAGES
from autogen import AssistantAgent, UserProxyAgent


# IMPORT OPENAI API KEY
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# CREATE THE AGENTS
# create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(name="assistant")
# create a UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(name="user_proxy")


# The assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""Hello, today you are my data analyst assistant and you should help me visualize data, make predictions, and explain your thinking.""",
)

# Originally in the documentation example: "Plot a chart of NVDA and TESLA stock price change YTD"
# Possible to modify the prompt and the program according to your needs



# Create a text completion request
response = oai.Completion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "api_base": "http://localhost:8000/v1",
            "api_type": "open_ai",
            "api_key": "NULL", # just a placeholder
        }
    ],
    prompt="Hi",
)
print(response)

# Create a chat completion request
response = oai.ChatCompletion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "api_base": "http://localhost:8000/v1",
            "api_type": "open_ai",
            "api_key": "NULL",
        }
    ],
    messages=[{"role": "user", "content": "Hi"}]
)





print(response)
