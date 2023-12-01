import os
import openai

from dotenv import load_dotenv, find_dotenv

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=1000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    print(response.usage)
    return response.choices[0].message["content"]


def message_template_1 (user_message_1):
    delimiter = "####"
    system_message = f"""
    Your task is to generate an overall summary using the user's input. \
    The user's input will be delimited by {delimiter} characters. \
    The output should be a text in UTF-8 format, written in Chinese. 
    """   
    messages =  [ 
        {'role':'system', 
         'content': system_message}, 
        {'role':'user',
         'content': f"{delimiter}{user_message_1}{delimiter}"}  
    ] 
    return messages


def message_template_2 (user_message_1, user_message_2):
    delimiter = "####"
    system_message = f"""
    Your task is to generate an overall summary using the previous summary plus user's new input. \
    This is an accumulative task. \
    The previous summary is enclosed within {delimiter} as shown below: {delimiter}{user_message_1}{delimiter} \

    Summarize the user's new input and incorporate it into the existing summary as the output. \
    Update the output to ensure its coherence. \
    The user's new input will be enclosed by {delimiter} characters. \
    The output should be a UTF-8 encoded text written in Chinese. \
    """   
    messages =  [ 
        {'role':'system', 
         'content': system_message}, 
        {'role':'user',
         'content': f"{delimiter}{user_message_2}{delimiter}"}  
    ] 
    return messages


def fetch_by_openapi(converted_subtitles):
    openai.api_key  = os.environ['OPENAI_API_KEY']
    for index, subtitle in enumerate(converted_subtitles):
        if (index ==0):
            messages = message_template_1(subtitle)
        else:
            messages = message_template_2(summaries, subtitle)
        summaries = get_completion_from_messages(messages)
    return summaries 
