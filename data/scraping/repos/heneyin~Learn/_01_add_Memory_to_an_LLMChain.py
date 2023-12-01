import env

from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

"""
最重要的步骤是正确设置 prompt。

我们有两个输入键：一个用于实际输入，另一个用于 Memory 类的输入。
重要的是，我们确保 PromptTemplate 和 ConversationBufferMemory 中的键匹配 (chat_history)。
"""

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

result = llm_chain.predict(human_input="Hi there my friend")
print("result1:", result)
print("-----------------------------------------")
result = llm_chain.predict(human_input="Not too bad - how are you?")
print("result2:", result)
print("-----------------------------------------")
result = llm_chain.predict(human_input="I'm fine, thank you, and you?")
print("result3:", result)
result = llm_chain.predict(human_input="你和openai 什么关系")
print("result4:", result)