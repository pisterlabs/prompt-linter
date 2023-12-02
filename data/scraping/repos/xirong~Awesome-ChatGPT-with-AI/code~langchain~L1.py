from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

openAI_chat = ChatOpenAI(temperature=0,request_timeout=100000,max_retries=3)

# result_single = openAI_chat([HumanMessage(content="Translate this sentence from English to Chinese. I love programming.")]).content
# print(result_single)

batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
        HumanMessage(content="Translate this sentence from English to Chinese. I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
        HumanMessage(content="Translate this sentence from English to Chinese. I love artificial intelligence.")
    ],
]
result_batch = openAI_chat.generate(batch_messages)
print(result_batch.llm_outputs)
print(result_batch.generations.pop().content)