import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]

def getList(input):
    delimiter = "```"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with \
    {delimiter} characters, and you will be given previous messages in the conversation. \
    When you receive a query, you need to understand what is being asked. \
    For example, you could receive a sentence like 'Hi, how are you? I'm looking for a 600ml microwave-safe bowl for cooking'. \
    In this query, you need to identify the important part, which is '600ml microwave-safe bowl for cooking'. \
    I want you to focus on the keywords that describe the need. \
    Once you have identified that, return a list, where each element is a string of what you analyzed. \
    Example of the list:
        ["600ml microwave-safe bowl for cooking", "250ml Glass Cup", "Handmade oven-safe dish"]\
            
    If no products or needs are found, generate an empty list.

    Only display the list of strings, nothing else.
    """
    messages =  [  
    {'role':'system', 
    'content': system_message},    
    {'role':'user', 
    'content': f"{delimiter}{input}{delimiter}"},  
    ] 
    list = get_completion_from_messages(messages)
    list = eval(list)
    print(list)
    return list
