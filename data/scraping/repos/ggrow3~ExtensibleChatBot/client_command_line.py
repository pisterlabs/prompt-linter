from chatbot_factory import ChatBotFactory
from chatbot_settings import ChatBotSettings
import os
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from colorama import Fore, Style, init

# from colorama import Fore, Style, init
init(autoreset=True)

chatbot_factory = ChatBotFactory()

selected_bot = ChatBotFactory().select_chatbot(ChatBotFactory.services, "Please select a chatbot from the following options:")

chatbot = chatbot_factory.create_service(selected_bot, ChatBotSettings(llm=ChatBotFactory().llms[ChatBotFactory().LLM_CHAT_OPENAI],memory=ConversationBufferMemory(), tools=['serpapi','wolfram-alpha']))

print(Fore.GREEN + "Please enter a prompt:")

while True:
    # Get user input from the command line
    user_input = input()
    
    # If the user types 'exit', end the chatbot session
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Get the chatbot's response using the get_bot_response_chat_completions method
    response = chatbot.get_bot_response(user_input)
    print(response)
    print(type(response))
    if(type(response).__name__ == "gTTS"):
       response.save("welcome.mp3")
       break
    elif(type(response) == "str"):
        # Print the chatbot's response
        print(Fore.GREEN + response)
    


