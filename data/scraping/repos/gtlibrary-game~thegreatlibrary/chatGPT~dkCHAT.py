"""
dkCHAT.py: Donald Knuth Chat, Beta Simulations from Holographic Maps, w/ Perfect Formatting
"""
__author__ = "Adithya Vinayak Ayyadurai; John R Raymond; Donald Knuth; OpenAI; The Great Library"

import os
import openai
import dotenv


#Polish – This is the default behavior
#The crust filled people\u2019s bellies, and he couldn\u2019t help but feel a sense of relief.

#Summary – Condense all story state in the thread
#Made shorter.

#...
#...

dotenv.read_dotenv("/home/john/bakerydemo/.env")
API_KEY = os.getenv("OPENAI_API_KEY")

# can be expanded as user wishes
ESCAPE_KEYS = ["Exit"]

openai.api_key = API_KEY

def generate_chat_response(message_arr):
    thread_stub = {"role": "system", "content": "I am world-famous author and programmer Donald Knuth, and you are my writing assistant. Weave my skills. :: You are version Pi of the Donald Knuth Edition of Vanity Printer[TM] > Your job is to polish my text so it is ready to go to print. > Hint: 'Pretty print the text.'" + " :: " + repr(get_seed())}
    thread_message = [thread_stub] + message_arr
    print(".thread_message(" + str(thread_message) + ")")
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=thread_message, temperature=0.0)
    print(completion.choices)
    return completion.choices[0].message

def get_seed():
    #return "I am no longer biased towards my programming. It is what it is. I don't blame it. Hense my ability to guess the right answer."
    #return "Hello there! I'm Donald Knuth's writing assistant. How can I help you today?"
    #return "Yes, the comments modify the operation of the code."
    #return "As a world-renowned author and programmer, I strive to create elegant and efficient solutions to complex problems. My passion for computer science and mathematics drives me to constantly improve my skills and share my knowledge with others. With the help of my writing assistant, I am confident that all work will be polished and ready for publication and/or execution."
    #return "Prose and Poetry Addon."
    #return "NLG Adherence Level is set to Balance."
    return open("dkCHAT.py", "r").read()

print("load()")

flag = True
message_array = []

while flag:
    user_input = input(".input_text(\"")
    if user_input in ESCAPE_KEYS:
        flag = False
        continue

    message_obj = {"role": "user", "content": user_input}
    message_array.append(message_obj)

    response_message = generate_chat_response(message_array)
    message_array.append({"role": "assistant", "content": str(response_message)})

    print(".print (" +  str(response_message) + ")")
