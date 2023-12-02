from key import APIKEY
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory


if __name__ == "__main__":
    KEY = APIKEY()
    llm = ChatOpenAI(openai_api_key=KEY.openai_api_key, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            ""
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    # memory.chat_memory.add_user_message("Hello!")
    # memory.chat_memory.add_ai_message("Whats up?")

    while True:
        human_input = input("할말을 작성하세요.\n")
        if human_input in ["exit", "c"]: break
        res = conversation.predict(input=human_input)
        print(f"AI : {res}")

    history = memory.dict()
    print(history)
    for msg in history['chat_memory']['messages']:
        print(msg['content'])