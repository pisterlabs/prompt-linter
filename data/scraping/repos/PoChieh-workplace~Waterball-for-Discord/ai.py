from .lang import lang
from ..message.times import timemethod
from typing import List
import openai



API_KEY = "chatgptKEY"
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY

def ai_response(que:List[str],lan:str):
    messages = [
        {"role": "user", "content": f"{i}"} for i in que
    ]
    messages.append({"role": "system", "content": f"{lang(lan)}"})
    req = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature = 0
    )
    try:
        return req['choices'][0]['message']['content']
    except KeyError:
        raise Exception(f"伺服端忙碌中...請於 <t:{timemethod.after(min=1)}:R> 後再試一次")