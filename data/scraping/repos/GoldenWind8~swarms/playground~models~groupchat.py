
from swarms import OpenAI, Flow
from swarms.swarms.groupchat import GroupChatManager, GroupChat

llm = OpenAI(
    openai_api_key="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",
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the flow
flow1 = Flow(
    llm=llm,
    max_loops=1,
    system_message="YOU ARE SILLY, YOU OFFER NOTHING OF VALUE",
    name='silly',
    dashboard=True,
)
flow2 = Flow(
    llm=llm,
    max_loops=1,
    system_message="YOU ARE VERY SMART AND ANSWER RIDDLES",
    name='detective',
    dashboard=True,
)
flow3 = Flow(
    llm=llm,
    max_loops=1,
    system_message="YOU MAKE RIDDLES",
    name='riddler',
    dashboard=True,
)
manager = Flow(
    llm=llm,
    max_loops=1,
    system_message="YOU ARE A GROUP CHAT MANAGER",
    name='manager',
    dashboard=True,
)


# Example usage:
agents = [flow1, flow2, flow3]

group_chat = GroupChat(agents=agents, messages=[], max_round=10)
chat_manager = GroupChatManager(groupchat=group_chat, selector = manager)
chat_history = chat_manager("Write me a riddle")