import openai
from dotenv import dotenv_values
import Common.aa_global_variable as global_var

secret = dotenv_values("./.env")
openai.api_key = global_var.load_var("OPENAI_API_KEY")

"""
    File name: aa_openai_function.py
    Description: This file contains functions to use openai api
"""

"""
    Description:
        Request openai api
    Parameters:
        content: string
        role: string
        model: string
        temperature: float
        max_tokens: int
        top_p: float
        frequency_penalty: float
        presence_penalty: float
    Return:
        chatgpt_content: string
"""
def openai_request(content, role="system", model="gpt-3.5-turbo", temperature=0.5, max_tokens=200, top_p=1, frequency_penalty=0, presence_penalty=0):
    try:
        response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
            "role": role,
            "content": content 
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
        )
        chatgpt_content = ""
        if(len(response['choices']) > 0):
            chatgpt_content = str(response['choices'][0]['message']['content']).replace("'", "''")
            if(chatgpt_content[0] == '"'):
                chatgpt_content = chatgpt_content.replace('"', "")
        return chatgpt_content
    except Exception as e:
        print(e)
        return e

"""
    Description:
        Use chatgpt to generate a tweet from a given article
    Parameters:
        article: string
    Return:
        response: string
"""
def article_to_tweet_chatgpt(article):
    try:
        return openai_request(content="Fait un tweet court et percutent en français en restant neutre à partir de l'article suivant : \n" + article)
    except Exception as e:
        print(e)
        return e
        
"""
    Description:
        Use chatgpt to generate a question from a given article
    Parameters:
        article: string
    Return:
        response: string
"""
def article_question_chatgpt(article):
    try:
        return openai_request("Pose une question ouverte sur le tweet suivant : \n\n" + article)
    except Exception as e:
        print(e)
        return e
    