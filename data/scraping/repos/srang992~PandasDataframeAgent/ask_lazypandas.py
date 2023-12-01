from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType


def ask_lazy_pandas(message, data, secret_key):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-0613",
            openai_api_key=secret_key,
        ),
        df=data,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    return agent.run(message)
