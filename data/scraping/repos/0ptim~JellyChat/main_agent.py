from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType, load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
import langchain

from tools.wiki_qa import wikiTool
from tools.defichainpython_qa import defichainPythonTool
from tools.ocean import oceanTools

from agent.prompt import PROMPT

load_dotenv()


def create_agent(memory, final_output_handler=None):
    print("🤖 Initializing main agent...")

    # Set debug to True to see A LOT of details of langchain's inner workings
    # langchain.debug = True

    # Create the main agent llm
    agent_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k-0613",
        temperature=0.7,
    )

    # Add the final output handler to the main agent llm
    if final_output_handler:
        agent_llm.callbacks = [final_output_handler]
        agent_llm.streaming = True

    # Create the llm for math separately so the streaming of the final response doesn't interfere with the main agent
    llm_for_math = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    tools = [wikiTool, defichainPythonTool] + load_tools(["llm-math"], llm=llm_for_math) + oceanTools

    system_message = SystemMessage(content=PROMPT)

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    open_ai_agent = initialize_agent(
        tools,
        agent_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
        max_iterations=4,
        early_stopping_method="generate",
    )

    return open_ai_agent


if __name__ == "__main__":
    memory = ConversationBufferMemory(
        memory_key="memory",
        return_messages=True,
    )
    local_agent = create_agent(memory)
    while True:
        question = input("⚡ Testing main agent: ")
        with get_openai_callback() as cb:
            response = local_agent(question)
            print(f"⚡ Output: {response['output']}")
            print(f"⚙ Total Tokens: {cb.total_tokens}")
            print(f"⚙ Prompt Tokens: {cb.prompt_tokens}")
            print(f"⚙ Completion Tokens: {cb.completion_tokens}")
            print(f"⚙ Total Cost (USD): ${cb.total_cost}")
