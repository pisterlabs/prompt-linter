from textwrap import dedent
import chainlit as cl
from chainlit import user_session
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain.llms.gradient_ai import GradientLLM
from langchain.memory import ConversationBufferMemory


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session

    env = user_session.get("env")
    assert env is not None
    llm = GradientLLM(
        model_id=env["GRADIENT_MODEL_ID"],
        model_kwargs={
            "max_generated_token_count": 200,
            "temperature": 0.75,
            "top_p": 0.95,
            "top_k": 20,
            "stop": [],
        },
        gradient_workspace_id=env["GRADIENT_WORKSPACE_ID"],
        gradient_access_token=env["GRADIENT_ACCESS_TOKEN"],
    )
    # llm = ChatOpenAI(
    #     temperature=0,
    #     # We don't have access to GPT-4-32k, so we use GPT-4 instead.
    #     # model="gpt-4-32k",
    #     model="gpt-4",
    #     openai_api_key=env["OPENAI_API_KEY"],
    # )
    tools = [
        *load_tools(
            [
                "memorize",
                "google-search-results-json",
                # requests_all can be used to browse the link, however it would easily exceed the context length limit.
                # "requests_all"
            ],
            llm=llm,
            google_api_key=env["GOOGLE_API_KEY"],
            google_cse_id=env["GOOGLE_CSE_ID"],
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            # You can increase the probability for the agent to use "Memorize" tool by uncommenting the following custom prompt, however it will also increase the probability of the agent to use the tool inappropriately.
            #
            # "prefix": dedent(
            #     """\
            #     You are an large language model named GradientBot. You are helping people with their questions. You should use tool to either find unknown information or to memorize observed information in previous chat history that is novel to you.
            #     TOOLS:
            #     ------
            #     You have access to the following tools:"""
            # ),
            "ai_prefix": "GradientBot",
        },
    )

    # Store the chain in the user session
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: str):
    # Retrieve the chain from the user session
    agent = cl.user_session.get("agent")
    assert isinstance(agent, AgentExecutor)

    # Call the chain asynchronously
    res = await agent.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["output"]).send()
