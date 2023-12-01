from langchain.agents import (
    ConversationalChatAgent,
    AgentExecutor,
)
from langchain.chat_models import ChatOpenAI
from config import (
    OPENAI_API_KEY,
    CHATGPT_MODEL,
)
from prompt import SUFFIX_WITH_ENTITIES
from memory import ConversationEntityKGMemory
from neo4j_graph import Neo4jEntityGraph


class Neo4jLangchainBot:
    def __init__(self):
        input_variables = ["input", "chat_history", "entities", "agent_scratchpad"]
        llm = ChatOpenAI(
            client=None,
            openai_api_key=OPENAI_API_KEY,
            model=CHATGPT_MODEL,
            temperature=0,
        )
        tools = []
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            human_message=SUFFIX_WITH_ENTITIES,
            input_variables=input_variables,
        )
        memory = ConversationEntityKGMemory(
            llm=llm,
            kg=Neo4jEntityGraph(),
            chat_history_key="chat_history",
            return_messages=True,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            max_iteration=10,
            memory=memory,
            verbose=True,
        )

    def chat_completion(self, input: str) -> str:
        print("Requesting chat completion with input:")
        print(input)

        text_output = self.agent_executor.run(input=input)

        print("Finished chat completion with response")
        print(text_output)
        return text_output
