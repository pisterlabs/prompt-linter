import json
import openai
import time

def get_key():
    with open('./key/OpenAPI_APIKEY.json') as json_file:
        json_data = json.load(json_file)
    return json_data

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
def getInfo(problem, code):
    OpenAIKEY = get_key()
    openai.api_key = OpenAIKEY['Key']['Authorization']    
    text = f'leetcode {problem}에 대해 설명해줘'
    _content = f"""
    your task is explain leetcode problem \"{problem}\" solution and evaluate user python code which is code below, delimited by triple backticks.
    
    how to solve problems with code.
    
    use korean.
    
    result format is Markdown.
    
    code : ```{code}```
    """
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": _content}])
    content = completion.choices[0].message.content
    return content
