from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import LLMSingleActionAgent, AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from agents.agent_prompts import PREFIX, SUFFIX, FORMAT_INSTRUCTIONS
from agents.agent_prompt_template import CustomPromptTemplate
from agents.agent_output_parser import CustomOutputParsers
from agents.tool.get_business_data import LLM_get_business_data, GetBusinessDataInput


# Set OpenAI LLM and embeddings
llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                      model='gpt-3.5-turbo-0613', client='')

# Set conversation memory buffer
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True)


# Set tools

tools = [
    Tool(
        name="get_business_data",
        func=LLM_get_business_data.run,
        description="A tool to get the business data information required by the user. The input is a string and must be a question about the business data.",
    )
]
tool_names = [tool.name for tool in tools]

# Set up prompt template
prompt = CustomPromptTemplate(
    prefix=PREFIX,
    instructions=FORMAT_INSTRUCTIONS,
    sufix=SUFFIX,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Set up output parser
output_parser = CustomOutputParsers()


# Set up the agent
llm_chain = LLMChain(llm=llm_chat, prompt=prompt)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)
business_faq_agent = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, max_iterations=3, verbose=True)

# agent_executor.run('atiende los domingos y puedo pagar en efectivo?')
