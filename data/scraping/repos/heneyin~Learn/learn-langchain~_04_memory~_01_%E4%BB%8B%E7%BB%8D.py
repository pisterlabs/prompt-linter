"""
默认下，Chain 和 agents 是无状态的。

当需要记住上下文，使用 Memory 来完成

LangChain提供两种形式的内存组件。
1. 首先，LangChain 提供了帮助实用程序来管理和操作以前的聊天消息。它们被设计为模块化且有用的，无论它们如何使用。
2. 其次，LangChain提供了将这些实用程序整合到链中的简单方法。


一般来说，对于每种类型的 memory，都有两种理解使用记忆的方法。
1. 从一系列消息中提取信息

我们将介绍最简单的内存形式：“缓冲”内存，它只涉及保留所有先前消息的缓冲区。
"""

"""
ChatMessageHistory
 saving Human messages, AI messages, and then fetching them all.
"""

import env

# 直接使用
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

print("history.messages:", history.messages)

"""
ConversationBufferMemory
 just a wrapper around ChatMessageHistory that extracts the messages in a variable.
"""
from langchain.memory import ConversationBufferMemory

# We can first extract it as a string.
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

result = memory.load_memory_variables({})
print("ConversationBufferMemory result:", result)


# We can also get the history as a list of messages

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

result = memory.load_memory_variables({})
print("ConversationBufferMemory result:", result)


"""
Using in a chain
"""

from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# 放入 memory
llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

result = conversation.predict(input="Hi there!")
print("ai result", result)

result = conversation.predict(input="Tell me about yourself.")
print("ai result", result)

"""
Saving Message History

This can be done easily by first converting the messages to normal python dictionaries, 
saving those (as json or something) and then loading those. Here is an example of doing that.
"""

import json

from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

dicts = messages_to_dict(history.messages)

print("saved dicts:", dicts)
