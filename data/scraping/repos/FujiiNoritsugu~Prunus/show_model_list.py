import os
import openai

# ファイルを開く
with open(os.getenv("OPENAI_API_KEY"), "r") as file:
    # ファイルからデータを読み込む
    data = file.read()

# 読み込んだデータを変数に設定
openai.api_key = data

models = openai.Model.list()
for model in models["data"]:
    print("-", model["id"])
