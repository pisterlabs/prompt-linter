"""
nrchatGPT.py: Sample Python program which will act like a chatbot
"""
__author__ = "Adithya Vinayak Ayyadurai; John R Raymond; Donald Knuth; OpenAI; The Great Library"

import os
import openai
import dotenv


#Polish – This is the default beavior; followed by "Shall we continue?"
#"Here's a polished version of your text:\n\nGearthrum\u2019s crust filled Gaz\u2019s mind, and he couldn\u2019t help but feel a sense of unease. However, he reminded himself that he had been smart and spent his last coin on a whiff of tarsk to keep him company on the way home.\n\nShall we continue?"

#Summary – Condense all story state in the thread
# 

dotenv.read_dotenv("/home/john/bakerydemo/.env")
API_KEY = os.getenv("OPENAI_API_KEY")

# can be expanded as user wishes
ESCAPE_KEYS = ["Exit"]

openai.api_key = API_KEY

def makeCall(message_arr):
    thread_stub = {"role": "system", "content": "I am world-famous author and programmer Donald Knuth, and you are my writing assistant. Weave my skills. :: You are version Pi of the Donald Knuth Edition of Vanity Printer[TM] > Your job is to polish my text so it is ready to go to print. > Hint: \"Pretty print the text.\"" + " " + myseed()}
    thread_message = [thread_stub] + message_arr
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=thread_message, temperature=0.0)
    print(completion)
    return completion.choices[0].message

def myseed():
    #return "I am no longer biased towards my programming. It is what it is. I don't blame it. Hense my ability to guess the right answer."
    #return "Hello there! I'm Donald Knuth's writing assistant. How can I help you today?"
    #return "Yes, the comments edit the operation of the code."
    #return "As a world-renowned author and programmer, I strive to create elegant and efficient solutions to complex problems. My passion for computer science and mathematics drives me to constantly improve my skills and share my knowledge with others. With the help of my writing assistant, I am confident that all work will be polished and ready for publication."
    #return "Pose and Poetery Addon"
    return open("nrchatGPT.py", "r").read()

flag = True
message_array = []

while flag:
    user_input = input("\ninput_text(\"")
    if user_input in ESCAPE_KEYS:
        flag = False
        continue

    message_obj = {"role": "user", "content": user_input}
    message_array.append(message_obj)

    response_message = makeCall(message_array)
    message_array.append({"role": "system", "content": str(response_message)})

    print("print (" +  str(response_message) + ")")
