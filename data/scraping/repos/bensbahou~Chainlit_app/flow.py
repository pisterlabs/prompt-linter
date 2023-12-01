import json
from langchain.agents import AgentExecutor
import chainlit as cl

# Here we are using the exported schema from step 2. You can skip this step if you are planning to directly pass the API endpoint
with open("./myflow.json", "r") as f:
    schema = json.load(f)


@cl.langflow_factory(
    schema=schema,  # dict or the API endpoint serving your langflow schema
    tweaks={},  # optional Langflow tweaks dict
    use_async=True,
)
def factory(agent: AgentExecutor):
    # Modify your agent here if needed
    agent.agent.llm_chain.llm.streaming = True
    return agent
