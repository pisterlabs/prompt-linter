import os
import sys
import openai


def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [
        {"role": "system", "content": "You are a shell command suggestor. \
         Reply with shell code only. No human text. \
         User asks what a shell command he needs to use for the task given.\
         If you cannot determine the shell command, reply with a 'Cannot assist with it.'"},
        {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model,
                                            messages=messages,
                                            temperature=0,  # this is the degree of randomness of
                                            )
    return response.choices[0]. message["content"]


if not 'OPENAI_API_KEY' in os.environ:
    print("Please set OpenAI API key as 'OPENAI_API_KEY' environment variable. https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key")
    exit()

openai.api_key = os.getenv("OPENAI_API_KEY")

if len(sys.argv) != 2:
    print("Usage: shell_assist.py 'YOUR_QUESTION'")
    exit()

user_prompt = sys.argv[1]
response = get_completion(user_prompt)
print(response)
