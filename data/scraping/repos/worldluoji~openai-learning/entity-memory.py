
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

llm = OpenAI()

entityMemory = ConversationEntityMemory(llm=llm)
conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=entityMemory
)

answer=conversation.predict(input="我叫张老三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货")
print(answer)

# 指定使用 EntityMemory。可以看到，在 Verbose 的日志里面，整个对话的提示语，多了一个叫做 Context 的部分，里面包含了刚才用户提供的姓名、订单号和邮箱。