# -*- coding: utf-8 -*-

import os
from openai import OpenAI


def keywords_to_str(keywords:list)->str:
    """Bekommt eine liste mit keywords und setzt diese einen String mit "," und zum Schluss "und" zusammen
        z.B. ['Hund','Katze', 'Maus'] -> 'Hund, Katze und Maus'

    Args:
        keywords (list): _description_

    Returns:
        str: _description_
    """    
    keyword_str = "" 
    for i, kword in enumerate(keywords):
        keyword_str += str(kword)
        if i < len(keywords) - 2:
            keyword_str += ","
        elif i == len(keywords) - 2: 
            keyword_str += " und "
    return keyword_str

def get_age_span_from_tuple(age:tuple)->str: 
    """ Bekommt einen tuple mit zwei Zahlen und liefert einen String zurück. 

    Args:
        age (tuple): _description_

    Returns:
        str: _description_
    """    
    if age[0]==age[1]: 
        return f"{age[0]}"
    else: 
        return f"{age[0]} bis {age[1]}"
    
def get_prompt_text(keywords:list, genre:str, age:tuple, wordlimit:int=750)->str:
    return f"Nimm die Rolle eines Kinderbuchautors an. Schreibe eine {genre} für {get_age_span_from_tuple(age)} jährige. Die Geschichte soll von {keywords_to_str(keywords)} handeln. Die Geschichte soll maximal {wordlimit} Wörter haben."

def get_story(client:OpenAI, promt:str)->str: 
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": promt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return (chat_completion.choices[0].message.content)


def get_text_for_keywords(client:OpenAI, keywords:list,  genre:str, age:tuple, wordlimit:int)->tuple[str, str]:
    prompt = get_prompt_text(keywords, genre, age, wordlimit)
    print(f"Promt: {prompt}")
    story = get_story(client, prompt)
    return story, prompt

