# a multi-round chat interface between FAROS and OpenAI's API
# Currently it works with gpt-3.5-turbo. 

import json
import openai
import sys
import os
from dotenv import load_dotenv

system_message ="""You are now an expert in the Clang/LLVM compiler. 
You are going to help me understand why it behaves 
differently when compiling the same input program's 
loop using two different sets of compiler options 
Option 1: sequential baseline 
Option 2: OpenMP enabled."""

# global messages
messages = [{"role": "system", "content": system_message}] 

load_dotenv() # load environment variables from the .env file

# Replace with your own API key
openai.api_key = os.getenv("OPENAI_API_KEY") #"your-api-key"

# Calling ChatGPT API: passing user msg, returning a response
# Also update the global messages list as the chat history
def getAnswer(msg: str) -> str:
    # append the user's message info as a dictionary to the msgs list
    messages.append({"role": "user", "content": msg})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens = 200,
        temperature=0.2
    )['choices'][0]['message']['content'].strip()
    # strip() removes the leading and trailing spaces

    # append the response info. as a dictionary into the msgs list
    messages.append({"role": "assistant", "content": response})

#    print("all messages:=======================")
#    print(messages)
    
    #TODO OpenAI has tiktoken to count tokens
    list_as_str = json.dumps(messages)
    tokens = list_as_str.split()
    token_count = len(tokens)
    # This checks if the length of the msgs list has exceeded some limit
    # and if so, removes the second and third elements of the list.
    # This ensures that the msgs list does not grow too large and consume too much memory.
    if len(messages) >= 11 or token_count > 2000:
        messages.pop(1)  # list index id starts from 0 , we keep the system message
        messages.pop(2)

    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python thisfile.py your prompt here")
        sys.exit(1)

    #prompt = sys.argv[1]
    prompt = ' '.join(sys.argv[1:])   
    
    
    response = getAnswer(prompt)    
    print(f"{response}")
