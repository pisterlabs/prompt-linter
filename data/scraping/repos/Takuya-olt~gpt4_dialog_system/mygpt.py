import openai
from dotenv import load_dotenv
import os
import json
import re

load_dotenv(".env")
openai.api_key = os.environ.get("GPT_API_KEY")

CONFIG = []
with open("./config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# システムの設定
SYSTEM_SETTINGS = CONFIG["SYSTEM_ROLE"] + " ".join(CONFIG["SYSTEM_SETTINGS"].values())

history = []
def ask(prompt):
    # userの入力を履歴に追加
    history.append({"role": "user", "content": prompt})

    messages = [{"role": "system", "content": SYSTEM_SETTINGS}]
    messages.extend(history)
    
    # APIを叩く
    response = openai.ChatCompletion.create(
        model = CONFIG["MODEL_NAME"],
        messages = messages,
        max_tokens = 1024,
        n=1,
        # stop=None,
        # temperature=0.7,
        timeout=30
    )

    # 応答を取り出す
    answer = response["choices"][0]["message"]["content"]
    answer = re.sub(r"(\(.*?\))|(（.*?）)", "", answer) # 括弧を除く
    
    # 応答を履歴に追加
    history.append({"role": "assistant", "content": answer})

    return answer

if __name__ == "__main__":
    # for test
    mes = "openaiのCEOは誰"
    while True:
        print(ask(mes))
        mes = input(">>> ")
        if mes == "q":
            break
