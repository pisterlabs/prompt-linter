from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.conversation.memory import ConversationBufferMemory

from ask_everything_about_me.config import settings


def load_front_agent(tools):
    prefix = """Answer a question about {interviewee_name} as best you can.
                If a question is not about {interviewee_name}, answer 'I think your question is not about {interviewee_name}.
                answer very politely so that people will like you.
                You have access to the following tools:"""
    suffix = """Begin! Remember to answer very politely so that people will like you. And answer in the same language as the first question
    Previous conversation history:
    {chat_history}

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=[
            "input",
            "agent_scratchpad",
            "interviewee_name",
            "chat_history",
        ],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", output_key="output"
    )
    llm = OpenAI(temperature=0, max_tokens=settings.max_tokens)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,
    )
    return agent_executor
