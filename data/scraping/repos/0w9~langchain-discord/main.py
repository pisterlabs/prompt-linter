from .langchain_discord import DiscordWebhookTool
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

import os
tools = [
    DiscordWebhookTool()
]

llm = OpenAI(temperature=0)

os.environ["WEBHOOK_URL"] = "https://discord.com/api/webhooks/1111194690939785296/azQ6OqeHyh17_XjIsv-sdnMpYePD5_lyHIKV_MRYOpWc_Qlnul_8W7b2JNBFkZmdx9yd"
os.environ["WEBHOOK_USERNAME"] = "LangChain Agent"

agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Send a joke to the webhook!")