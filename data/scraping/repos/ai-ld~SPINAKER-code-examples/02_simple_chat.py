# Loading neccesary libraries
import openai
import os
from dotenv import load_dotenv

# loading OpenAI API KEY
# openai.api_key = "your_API_KEY"
load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')

# Specifying initial prompt (instructions for the chatbot on how to behave)
prompt = "The following is a conversation with an AI assistant named Chatty. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\n"

# Creating chat_log variable for collecting history of messages
chat_log = ""

# definition of query function that will be used to send messages to GPT-3 and receive its replies
def query(model_input):
    """
    Method that takes Human messages as an input <<model_input>> and generates and API call to generate completion (model's answer)
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=model_input,
        temperature=0.8,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=1,
        stop=['\nHuman']
        )
    answer = response.choices[0].text.strip()
    return answer

# Printing initial message to the user
print("AI: Hello! I am Chatty, your friendly chatbot. How can I help you?")

# Loop responsible for conversation in the Terminal
while True:
    # capturing message entered by the user in the terminal
    user_input = "Human: " + input("Human: ") + "\n"

    # when user types letter 'q' the chat is stopped
    if user_input == 'Human: q\n':
        print('***exiting the chat***')
        break

    # creating input for the model. It contains: initial prompt (instructions), chat log(previos messages) and latest user input
    model_input = prompt + chat_log + user_input

    # generating output of the model and assigning it to the 'model_output_ variable
    model_output = query(model_input)

    # adding latest message of user and latest response of GPT-3 to the chat_log of the conversation
    chat_log += user_input + model_output

    # priting the message of chatbot in the terminal
    print(model_output)


