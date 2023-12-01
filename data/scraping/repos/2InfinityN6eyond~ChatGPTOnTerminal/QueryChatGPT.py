import os
import glob
import json
import time
import openai

# https://platform.openai.com/docs/models/overview
GPT_MODEL_NAME = "gpt-4"
OPENAI_API_KEY_FILE_PATH = "./openai_api_keys.json"

if __name__ == "__main__" :
    with open(OPENAI_API_KEY_FILE_PATH, "r") as fp :
        openai_api_key = json.load(fp)[0]
        openai.api_key = openai_api_key

    while True :
        question = input("Enter question (q to quit) : ")
        if question in ["Q", "q"] :
            break

        # openAI API 호출
        messages = [{
            "role" : "user", 
            "content" : question
        }]
        response = openai.ChatCompletion.create(model=GPT_MODEL_NAME, messages=messages)
        resp_content = response['choices'][0]['message']['content']
        print(">>")
        print(resp_content)
        print("-------------------------------")
