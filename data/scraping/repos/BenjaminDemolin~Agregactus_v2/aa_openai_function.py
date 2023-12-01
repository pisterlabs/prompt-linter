import openai
from dotenv import dotenv_values

secret = dotenv_values("./.env")
openai.api_key = secret["OPENAI_API_KEY"]

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
        # return openai_request(content="Fait un tweet percutent en français en restant neutre à partir de l'article suivant : \n" + article)
    except Exception as e:
        print(e)
        return e

"""
    Description:
        Ask to chatgpt which tweet is the best in a list of tweets
    Parameters:
        tweet_list: list
        tweet_size: int
    Return:
        tweet: array tweet infos
                -1: Rate limit reached
                -2: No valid tweet
                -3: No valid tweet number
"""
def get_best_tweet(tweet_list, tweet_size=280):
    try:
        print("---ask openai for best tweet---")
        # Number of tweet in the list with a size inferior to max size
        nb_tweet_valid = 0
        # Index of the only tweet with a size inferior to max size if there is only one
        index_only_one_tweet = -1
        # Body of the openai request
        body = "Retourne juste le numéro du tweet le plus important parmi les suivants : \n\n"
        for i, tweet in enumerate(tweet_list):
            openai_tweet = tweet[2]
            # if tweet is inferior to max size then add it to the body
            if(len(openai_tweet) < tweet_size):
                body += str(i) + " : " + openai_tweet + "\n\n"
                nb_tweet_valid += 1
                index_only_one_tweet = i
        # if no tweet is inferior to max size then return false
        if(nb_tweet_valid == 0):
            return -2
        # if only one tweet is inferior to max size then return it
        if(nb_tweet_valid == 1):
            return tweet_list[index_only_one_tweet]
        # else ask openai for the best tweet
        tweet_number_str = openai_request(body,temperature=0,max_tokens=100)
        tweet_number_int = -3
        # if openai return a valid tweet number then return the tweet (not Rate limit reached)
        if(isinstance(tweet_number_str, str)):
            print("openai best tweet : " + tweet_number_str)
            for i in range(len(tweet_list), 0, -1):
                if(str(i) in tweet_number_str):
                    tweet_number_int = i
                    break
            # if no int found in the openai response then return -3
            if(tweet_number_int == -3):
                return -3
            # else return the tweet infos
            else:
                return tweet_list[tweet_number_int - 1]
        else:
            # if openai return an error then return -1
            return -1
    except Exception as e:
        print(e)
        return e