# %%
# https://platform.openai.com/docs/api-reference/completions
# ChatGPT Completions
import os
import json
import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    # Open the JSON file for reading
    with open("./Config/api.json", "r") as json_file:
        data = json.load(json_file)
        openai.api_key = data["openai2"]
except FileNotFoundError:
    print(f"The file api.json does not exist.")

str1 = openai.Completion.create(
    # model="gpt-3.5-turbo-instruct",
    model="text-davinci-001",
    prompt="說一個黃色笑話",
    max_tokens = 2000
    # temperature=0
)
print(str1["choices"][0]["text"])
# %%
# ChatGPT Completions
import os
import json
import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    # Open the JSON file for reading
    with open("./Config/api.json", "r") as json_file:
        data = json.load(json_file)
        openai.api_key = data["openai"]
except FileNotFoundError:
    print(f"The file api.json does not exist.")

full_q = ""
while True:

    q = input("請輸入你的問題: ")
    full_q = full_q + " \n " + q

    str1 = openai.Completion.create(
        model = "gpt-3.5-turbo-instruct",
        # model = "text-davinci-001",
        prompt = full_q,
        max_tokens = 2000
        # temperature=0
    )
    print(str1["choices"][0]["text"])
    break

# %%
# ChatGPT Chat
import os
import json
import openai
try:
    # Open the JSON file for reading
    with open("./Config/api.json", "r") as json_file:
        data = json.load(json_file)
        openai.api_key = data["openai"]
except FileNotFoundError:
    print(f"The file api.json does not exist.")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a cat"},
       # {"role": "assistant", "content": "喵喵"},
        {"role": "user", "content": "請你用貓語自我介紹一下"}
        ]
)

print(completion.choices[0].message.content)

# %%
# Audio Create Translation
# https://platform.openai.com/docs/api-reference/audio
import os
import json
import openai
try:
    # Open the JSON file for reading
    with open("./Config/api.json", "r") as json_file:
        data = json.load(json_file)
        openai.api_key = data["openai"]
except FileNotFoundError:
    print(f"The file api.json does not exist.")
# 聲音轉文字
audio_file = open("./Data/out.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript["text"])
# 聲音轉英文
translate = openai.Audio.translate("whisper-1", audio_file)
print(translate["text"])
# %%
