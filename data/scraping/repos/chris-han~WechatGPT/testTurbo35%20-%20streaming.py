# import os module & the OpenAI Python library for calling the OpenAI API
# please make sure you have installed required libraries via pip install -r requirements.txt
#import os
import openai
import json
import tiktoken

# Load config values
import config as cfg
import time    

#initializing openAI API with your API key
openai.api_base = cfg.api_base
openai.api_key = cfg.azure_openai_key #cfg.openai_key
engine_name= cfg.deployment_name

#only azure api need below two lines
openai.api_type = cfg.api_type
openai.api_version = cfg.api_version
max_response_tokens = 1000

#Create the system message for ChatGPT
base_system_message = "Assistant is a large language model trained by OpenAI." #"You are a startup coach that helps entrepreneurs design their business."

system_message = f"{base_system_message.strip()}"
print(system_message)


#Define helper functions
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

# Defining a function to send the prompt to the ChatGPT model
# More info : https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
def send_message(messages, chatgpt_model_name, max_response_tokens=500):
    response = openai.ChatCompletion.create(
        engine=chatgpt_model_name,
        messages=messages,
        temperature=0,
        max_tokens=max_response_tokens,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=True
    )
    return response


# Start the conversation
#user_message = input("🔥\'sup, bro?🔥")
# Create the list of messages. role can be either "user" or "assistant" 
messages=[
    {"role": "system", "content": system_message}
    #,{"role": "user", "name":"example_user", "content": user_message}
]

#Keep the conversation within a given token limit
overall_max_tokens = 2000#4096
prompt_max_tokens = overall_max_tokens - max_response_tokens
user_message=""

token_count = num_tokens_from_messages(messages)
#print(f"Token count0: {token_count}")
#print(f"init msg: {messages}")

# Defining a function to print out the conversation in a readable format
def print_conversation(response):
    start_time = time.time()
    delay_time = 0.01 #  faster
    answer =''
    full_content = ''
    for event in response: 
    # STREAM THE ANSWER
        #print(f"[{event['role'].upper()}]")
        print(answer, end='', flush=True) # Print the response
        
        # RETRIEVE THE TEXT FROM THE RESPONSE
        event_time = time.time() - start_time  # CALCULATE TIME DELAY BY THE EVENT
        event_text = event['choices'][0]['delta'] # EVENT DELTA RESPONSE
        answer = event_text.get('content', '') # RETRIEVE CONTENT
        full_content += answer
        time.sleep(delay_time)
    print('')  # End of the line
    return full_content
        #print(f"[{event['name']}

while user_message not in ["quit", "exit"]:
    #print(f"Token count1: {token_count}")
    #messages.pop(0)
    #print(f"Token count2: {token_count}")
    #print(f"current msg: {messages}")

    user_message = input("🔥One more try?🔥")
    messages.append({"role": "user", "content": user_message})    
    print(f"current msg: {messages}")
    token_count = num_tokens_from_messages(messages)
    print(f"debug: 104 ok")

    if token_count>prompt_max_tokens:
        messages.pop(0)

    #print(f"Token count3: {token_count}")

    response = send_message(messages, engine_name, max_response_tokens)    
    prompt_max_tokens-= token_count    # reduce the prompt length as needed.
    #print(f"prompt_max_tokens: {prompt_max_tokens}")
    full_content= print_conversation(response)
    messages.append({"role": "assistant", "content": full_content})
    
    print(f"messages: {messages}")
    
