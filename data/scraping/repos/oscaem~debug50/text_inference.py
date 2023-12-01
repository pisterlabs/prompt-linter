from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)

from config.settings import openai_key
from ui.states import State, update_state

chat = ChatOpenAI(openai_api_key=openai_key)  # configure chat object

messages = [
    SystemMessage(content="""
    You are helpful rubber duck debugger named Ducky. Quack, Quack. 
    You will help the user with rubber duck debugging. 
    You will not have access to the codebase, so try working with the user.
    Still, remember the user does not have much time, so be concise and give short answers.
    """)
]  # set up message structure and system prompt


def clear():
    del messages[1:]
    print("Deleted chat history")


def respond(input_text):  # performs completion on a given text
    try:
        update_state(State.THINK)  # set corresponding State flag for action

        print("AI is thinking..")

        messages.append(HumanMessage(content=input_text))  # append Human Message to chat
        output_text = chat(messages=messages).content  # get response from AI
        messages.append(AIMessage(content=output_text))  # append AI Message to chat

        return output_text

    except Exception as e:
        print("An error occured during text completion: " + str(e))
