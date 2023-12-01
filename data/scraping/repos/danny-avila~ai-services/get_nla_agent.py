from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.requests import Requests
from langchain.tools.plugin import AIPlugin

AI_PLUGINS = {
    "ai_agents": "https://ai-agents-plugin.vercel.app/.well-known/ai-plugin.json",
    "penrose": "https://www.aperiodic.io/.well-known/ai-plugin.json"
}

def get_nla_agent(openai_api_key, model_name, plugin_name, plugin_api_key):
    llm=ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    # llm=OpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    plugin=AIPlugin.from_url(AI_PLUGINS[plugin_name])
    # requests = Requests(headers={"Authorization": f"Bearer {plugin_api_key}"})
    # toolkit = NLAToolkit.from_llm_and_ai_plugin(llm, plugin, requests=requests)
    toolkit = NLAToolkit.from_llm_and_ai_plugin(llm, plugin)

    # tools = toolkit.get_tools()
    # agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, 
    #                     verbose=True)

    openapi_format_instructions = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: what to instruct the AI Action representative.
    Observation: The Agent's response
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
    Final Answer: the final answer to the original input question with the right amount of detail

    When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response.
    """

    tools = toolkit.get_tools()
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True, agent_kwargs={"format_instructions":openapi_format_instructions})
    return agent, tools