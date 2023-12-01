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
    #   HumanMessage(content='在吗？', additional_kwargs={}, example=False),
    #   AIMessage(content='有什么事?', additional_kwargs={}, example=False)
    # ]


# 使用 ConversationChain 和 OpenAI 生成回复
def use_conversation_chain():
    llm = OpenAI(model_name="text-davinci-002", n=2, temperature=0.3)
    conversation = ConversationChain(llm=llm, verbose=True)
    conversation.predict(input="小明有1只猫")
    conversation.predict(input="小刚有2只狗")
    # conversation.predict(input="小明和小刚一共有几只宠物?")
    result = conversation.run("小明和小刚一共有几只宠物?")
    print(result)


# 使用 ChatMessageHistory + messages_to_dict 长期保存历史消息
def use_chat_message_history_with_messages_to_dict():
    history = ChatMessageHistory()
    history.add_user_message("在吗？")
    history.add_ai_message("有什么事?")
    print(history.messages)
    # [
    #   HumanMessage(content='在吗？', additional_kwargs={}, example=False),
    #   AIMessage(content='有什么事?', additional_kwargs={}, example=False)
    # ]
    history_dict = messages_to_dict(history.messages)
    print(history_dict)
    # [
    #   {'content': '在吗？', 'additional_kwargs': {}, 'example': False, 'type': 'human'},
    #   {'content': '有什么事?', 'additional_kwargs': {}, 'example': False, 'type': 'ai'}
    # ]

    new_messages = messages_from_dict(history_dict)
    print(new_messages)
    # [
    #   HumanMessage(content='在吗？', additional_kwargs={}, example=False),
    #   AIMessage(content='有什么事?', additional_kwargs={}, example=False)
    # ]


if __name__ == '__main__':
    # use_chat_message_history()
    # use_conversation_chain()
    use_chat_message_history_with_messages_to_dict()
