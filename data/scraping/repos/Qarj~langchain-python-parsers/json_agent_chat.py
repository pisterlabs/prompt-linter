import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from Tools.JobSearchTool import JobSearchTool
from JsonParser.JsonPromptTemplate import JsonPromptTemplate
from JsonParser.JsonOutputParser import JsonOutputParser

# https://python.langchain.com/docs/modules/model_io/models/llms/integrations/azure_openai_example

# See here for Azure: https://api.python.langchain.com/en/latest/modules/chat_models.html

search = JobSearchTool()
tools = [
    Tool(
        name="JobSearchTool",
        func=search.run,
        description="Useful for searching for jobs, input is jobTitle and jobLocation as a JSON object",
    )
]

prompt = JsonPromptTemplate(
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)

# deployment_name = os.getenv("OPENAI_API_DEPLOYMENT_NAME")

output_parser = JsonOutputParser()
llm = ChatOpenAI(temperature=0, model="gpt-4")
# llm = AzureChatOpenAI(temperature=0, deployment_name=deployment_name)
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

result = agent_executor.run("Search me test engineer jobs in London.")
print("\n\n")
print(result)
