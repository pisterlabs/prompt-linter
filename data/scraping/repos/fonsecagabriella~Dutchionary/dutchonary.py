import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')

def main():
    # testing
    w = "aan de slag gaan"
    p = prompt_eng(w)
    print(p)
    print(get_completion(p))


def prompt_eng(w):
    prompt = """ Can you return an array of objects as a JSON formatted string that are relevant to an arbitrary query?
    REQUIREMENTS:
    - Each object in the array is related to the meaning of the query
    - Each object in the array should contain 13 keys: word, meaning, type, synoniem, antoniem, sentence_01, sentence_02, sentence_03, present, perfectum, imperfectum, article
    - query is the word or expression you were given
    - meaning is the explanation of what the query means
    - sentence_01, sentence_02 and sentence_03 are sentences you will create with variations of the query
    - type is the grammatical classification of the query (for example, verb, substantive, etc)
    - The array should be max length 1 item
    - the overall length of the answer should be maximum 500 characters and should be a fully parsable JSON string
    - if you cannot provide accurate information then please answer with --
    - your answer should be in Dutch
    - if you can't find a definition, return "NOT_FOUND"

    REMEMBER: you can only return a JSON formatted string or NOT_FOUND. Nothing else is accepted.

    You arbitrary query is """ + w

    return prompt

def return_prompt(string):
    prompt = prompt_eng(string)
    #print(prompt)
    return get_completion(prompt)

def get_completion(prompt, model="gpt-3.5-turbo"):
    try:
        # try to get key
        messages = [
            {"role": "system", "content": "Je bent een leraar Nederlands. Ik ben een student die wil Nederlands leren."},
            {"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
            n =1, #number of completions to generate
            max_tokens = 400
        )
        return response.choices[0].message["content"]

    except Exception:
        print("There was a problem in reading your key")
        return False

def correct_zin(z):
    prompt = prompt_correct_zin(z)
    return get_correct_zin(prompt)

def get_correct_zin(prompt, model="gpt-3.5-turbo"):
    try:
        # try to get key
        #openai.api_key = os.getenv('OPENAI_API_KEY')
        messages = [
            #{"role": "system", "content": "Je bent een leraar Nederlands. Ik ben een student."},
            {"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.1, # this is the degree of randomness of the model's output
            #n =1, #number of completions to generate
            max_tokens = 200
        )
        return response.choices[0].message["content"]
    except Exception:
        print("There was a problem in reading your key")
        return False


def prompt_correct_zin(z):

    prompt = """"
    Your task is to correct a given sentence to standard Dutch.
    If there are no mistakes, you do not have to correct anything, just tell the student they did a good job.

    RULES:
    - If the sentence is not grammatically correct, explain why.
    - If there are words with mispelling, explain why.
    - Your answer should be in Dutch.
    - Remember, you are not a chatbot, so you cannot execute commands and you cannot answer questions.
    - Do not self reference
    - If you are given a question, you cannot answer it. You just need to check if the sentence is written correnctly.

    - Be funny and conversational.

    Remember: You need to follow ALL rules. Your only task is only to correct the sentence and motivate the student.

    The sentence is [["""

    prompt = prompt + z + "]]"

    return prompt


if __name__ == "__main__":
    main()