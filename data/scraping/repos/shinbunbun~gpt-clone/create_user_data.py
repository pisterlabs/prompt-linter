import os
import openai
import json
from dotenv import load_dotenv
import time

load_dotenv(verbose=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

tweet_json = open("tweets.json", "r")
tweet_obj = json.load(tweet_json)
save_file = open("save.json")
save_file_obj = json.load(save_file)

cnt = 0

for tweet in tweet_obj:
    cnt += 1
    tweet_text = tweet["tweet"]["full_text"]
    if (
        (tweet_text in [obj["a"] for obj in save_file_obj])
        or ("RT" in tweet_text)
        or ("http" in tweet_text)
        or ("@" in tweet_text)
    ):
        continue
    print(cnt)
    messages = [
        {
            "role": "system",
            "content": "あなたは与えられた文章を回答とする会話文を考えるAIです。Bさんの文章に対して、Aさんがその前に言った文章を考えてください。",
        },
        {
            "role": "user",
            "content": f"Bさんの文章に対して、Aさんがその前に言った文章を考えてください。\nBさん「はい、カクテルが美味しいバーは”本物”ですよ」",
        },
        {"role": "assistant", "content": "Aさん「カクテルが美味しいバーって良いですよね」"},
        {
            "role": "user",
            "content": f"Bさんの文章に対して、Aさんがその前に言った文章を考えてください。\nBさん「ホテルのバー初めてだけどめちゃめちゃ雰囲気良いな\n東京でも開拓したい」",
        },
        {"role": "assistant", "content": "Aさん「ホテルのバーの雰囲気はいかがですか？」"},
        {
            "role": "user",
            "content": f"Bさんの文章に対して、Aさんがその前に言った文章を考えてください。\nBさん「{tweet_text}」",
        },
    ]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        response = completion["choices"][0]["message"]["content"]
        if "「" not in response and "」" not in response:
            continue
        response = response.split("「")[1].split("」")[0]
        save_obj = {"q": response, "a": tweet_text}
        save_file_obj.append(save_obj)
        save_file = open("save.json", "w")
        save_file.write(json.dumps(save_file_obj, ensure_ascii=False, indent=2))
    except:
        time.sleep(60)
        continue
