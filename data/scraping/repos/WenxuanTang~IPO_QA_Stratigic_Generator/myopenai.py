# my_module.py

import openai
import nltk

def get_completion_from_messages(messages, 
                                 temperature,
                                 model="gpt-3.5-turbo-0613",
                                 max_tokens=1000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

def format_response(response):
    # Tokenize sentences using NLTK library
    sentences = nltk.sent_tokenize(response)
    
    # Add line breaks between each sentence
    formatted_response = '\n'.join(sentences)
    
    return formatted_response


#  model="gpt-3.5-turbo",
