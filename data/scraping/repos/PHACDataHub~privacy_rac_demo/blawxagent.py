from dotenv import load_dotenv
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agent.tools.ontology import ontology_tool
from agent.tools.interview import PAInterview
import os
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
#langchain.debug = True

load_dotenv()
openai_api_key=os.environ['OPENAI_API_KEY']

# Because we are using functions, we need to use model gpt-4-0613
llm=ChatOpenAI(openai_api_key=openai_api_key,temperature=0, model="gpt-4-0613")

tools = [ontology_tool,PAInterview()]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs=agent_kwargs, memory=memory)
