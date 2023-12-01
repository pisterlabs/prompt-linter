import openai_async
import json
import os
import asyncio
import time
def history_write(idlist, textprompt, textresponse):
    history = []
    with open(f"{idlist}.txt", "a") as f:
        textprompt = str(textprompt)
        textresponse = str(textresponse)
        user_dict = {"role": "user", "content": textprompt}
        response = {"role": "assistant", "content": textresponse}
        f.write(str(user_dict) + "\n")
        f.write(str(response) + "\n")

def get_history(idlist):
    if not os.path.exists(f"{idlist}.txt"):
        f = open(f"{idlist}.txt", "w+")
        f.close()
        print("No history detected, creating null")
    else:
        with open(f"{idlist}.txt", "r") as f:
            file_content = f.read()
            return file_content.splitlines()
async def question_answer(text, userid, token):
    history_list = get_history(userid)
    xe = [{"role": "system", "content": "You are Open-NeroGPT - NeroGPT with opened source code, based on GPT-3. Your developer is 'csoftware' - creator and founder of 'ArmaniOS Foundation' and 'AriGPT'. Be friendly and polite. Link for Open-NeroGPT: https://github.com/csoftware-arigpt/Open-NeroGPT"}]
    try:
        for i in range(0, len(history_list)):
            xe.append(eval(history_list[i]))
        xe.append({"role": "user", "content": text})
    except:
        xe = [{"role": "system", "content": "You are Open-NeroGPT - NeroGPT with opened source code, based on GPT-3. Your developer is 'csoftware' - creator and founder of 'ArmaniOS Foundation' and 'AriGPT'. Be friendly and polite. Link for Open-NeroGPT: https://github.com/csoftware-arigpt/Open-NeroGPT"}, {"role": "user", "content": text}]
    response = await openai_async.chat_complete(token, timeout=1000000, payload={"model": "gpt-3.5-turbo", "messages": xe,},)
    print(response.json())
    try:
            answer = response.json()["choices"][0]["message"]["content"]
            history_write(userid, text, answer)
            return answer
    except:
            print("error detected")
            with open(f"{userid}.txt",'r+') as file:
                    file.truncate(0)
            xe = [{"role": "system", "content": "You are Open-NeroGPT - NeroGPT with opened source code, based on GPT-3. Your developer is 'csoftware' - creator and founder of 'ArmaniOS Foundation' and 'AriGPT'. Be friendly and polite. Link for Open-NeroGPT: https://github.com/csoftware-arigpt/Open-NeroGPT"}, {"role": "user", "content": text}]
            response = await openai_async.chat_complete(token, timeout=1000000, payload={"model": "gpt-3.5-turbo", "messages": xe,},)
            answer = response.json()["choices"][0]["message"]["content"]
            history_write(userid, text, answer)
            return answer
async def wipedialog(userid):
                with open(f"{userid}.txt",'r+') as file:
                    file.truncate(0)
