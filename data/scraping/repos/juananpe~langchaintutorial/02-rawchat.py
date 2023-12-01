from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI

from langchain.schema import (
	SystemMessage,
	HumanMessage,
	AIMessage
)

load_dotenv()

chat = ChatOpenAI(temperature=.3)

messages = [
    SystemMessage(content="You are a helpful assistant"),
]

print("Hello! How can I help you?")

'''
## Second part  

while True:
    user_input = input("> ")
    # print("you sent: ", user_input)

    messages.append(HumanMessage(content=user_input))
    ai_response = chat(messages)
    messages.append(AIMessage(content=ai_response.content))
    print("AI: ", ai_response.content)

    ## third part
    print("\n\nHistory: ", messages, "\n")
    '''
