from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat
)
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
