from langchain.memory import ChatMessageHistory
from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.schema import messages_from_dict, messages_to_dict


# 使用 ChatMessageHistory 记录聊天历史
def use_chat_message_history():
    history = ChatMessageHistory()
    history.add_user_message("在吗？")
    history.add_ai_message("有什么事?")
    print(history.messages)
    # [
    # HumanMessage(content='在吗？', additional_kwargs={}, example=False),
    # AIMessage(content='有什么事?', additional_kwargs={}, example=False)
    # ]


if __name__ == '__main__':
    use_chat_message_history()
