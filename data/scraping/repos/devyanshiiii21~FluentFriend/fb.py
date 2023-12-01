import openai
import __api
import os

openai.api_key = __api.api

onlyfiles = [f for f in os.listdir('/Users/coding/Documents/vs/FluentFriend/model/dataset/input')]

def FB(input):
    model_engine = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=model_engine, 
        messages=[{"role":"system","content":"Rank the user input I have provided on a scale of 1-10 in the areas of [clarity, language quality, spelling, grammar] in a table. Also provide some tips for user to improve their skills upon."},{"role": "user", "content": input}], 
        max_tokens = 400, 
        stop=None, 
        temperature=0.1 
    )
    message = response.choices[0].message.content
    return message.strip()

def user_query(prompt):
    user_prompt = (f"{prompt}")
    result = FB(user_prompt)
    return result

def runner_4():
    try:
        for c in onlyfiles:
            cc = c.replace(".txt","")
            text = open('/Users/coding/Documents/vs/FluentFriend/model/dataset/input/'+c)
            text = text.readlines()
            pf = user_query(text)

            f = open('/Users/coding/Documents/vs/FluentFriend/model/dataset/feedback/'+cc+'_fb.txt', "x")
            f = open('/Users/coding/Documents/vs/FluentFriend/model/dataset/feedback/'+cc+'_fb.txt', "w")
            f.writelines(pf)
            f.close()
    except:
        pass

