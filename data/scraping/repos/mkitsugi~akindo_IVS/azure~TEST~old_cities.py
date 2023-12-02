import json
import os
import sys

import openai
from dotenv import load_dotenv

# APIキーの設定
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 引数で質問を受ける
question = sys.argv[1]


# 関数の実装
def get_belonging_prefecture(cities):
    """市町村名の羅列を受け取って、それぞれと都道府県の対応情報を持ったdictを返す。
    ただし特に定義のないものはモデルが学習した知識をそのまま使えるように「変更なし」を格納する"""

    def get_prefecture(city):
        return {"町田市": "神奈川県"}.get(city, "変更なし")

    prefecture_answer = [{"市区町村": city, "都道府県": get_prefecture(city)} for city in cities.split(",")]
    return json.dumps(prefecture_answer)


# AIが使うことができる関数を羅列する
functions = [
    # AIが、質問に対してこの関数を使うかどうか、
    # また使う時の引数は何にするかを判断するための情報を与える
    {
        "name": "get_belonging_prefecture",
        "description": "所属都道府県の変更情報を得る",
        "parameters": {
            "type": "object",
            "properties": {
                # cities引数の情報
                "cities": {
                    "type": "string",
                    "description": "市区町村名入力。半角カンマ区切りで複数要素を入力可能。各要素は「xx市」「xx区」「xx町」「xx村」のいずれか。例: 世田谷区,大阪市,府中町,山中湖村",
                },
            },
            "required": ["cities"],
        },
    }
]

# 1段階目の処理
# AIが質問に対して使う関数と、その時に必要な引数を決める
# 特に関数を使う必要がなければ普通に質問に回答する
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": question},
    ],
    functions=functions,
    function_call="auto",
)
print(json.dumps(response), file=sys.stderr)

message = response["choices"][0]["message"]
if message.get("function_call"):
    # 関数を使用すると判断された場合

    # 使うと判断された関数名
    function_name = message["function_call"]["name"]
    # その時の引数dict
    arguments = json.loads(message["function_call"]["arguments"])
    print("function_name: ", function_name)
    print("arguments: ", arguments)
    # 2段階目の処理
    # 関数の実行
    function_response = get_belonging_prefecture(
        cities=arguments.get("cities"),
    )
    print(function_response, file=sys.stderr)

    # 3段階目の処理
    # 関数実行結果を使ってもう一度質問
    second_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": question},
            message,
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            },
        ],
    )

    print(json.dumps(second_response), file=sys.stderr)
    print(second_response.choices[0]["message"]["content"].strip())
