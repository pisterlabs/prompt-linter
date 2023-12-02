"""

https://python.langchain.com/docs/modules/chains/how_to/memory


"""

import env
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(temperature=0.9)

# 记忆功能，存储对话上下文
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

result = conversation.run("Answer briefly. What are the first 3 colors of a rainbow?")
print("result 1: ", result)
# -> The first three colors of a rainbow are red, orange, and yellow.
result = conversation.run("And the next 4?")
# -> The next four colors of a rainbow are green, blue, indigo, and violet.
print("result 2: ", result)