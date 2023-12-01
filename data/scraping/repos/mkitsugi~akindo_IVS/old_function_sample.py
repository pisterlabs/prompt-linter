import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
query = "みかん、ぶどう、バナナについて、在庫が0であるか調べ、在庫が0の場合は商品のサプライヤーに追加注文のメールを送ってください。"

stock = [
    {"item": "みかん", "stock": 0, "supplier_for_item": "温州コーポレーション"},
    {"item": "りんご", "stock": 10, "supplier_for_item": "ハローKiddy Industory"},
    {"item": "バナナ", "stock": 0, "supplier_for_item": "The Donkey Foods"},
    {"item": "パイナップル", "stock": 1000, "supplier_for_item": "ペンパイナッポー流通"},
    {"item": "ぶどう", "stock": 100, "supplier_for_item": "グレープ Fruits inc."},
]


# 在庫チェック関数
def inventory_search(arguments):
    # 名前で在庫を探す
    inventory_names = json.loads(arguments)["inventory_names"]
    inventories = []
    for x in inventory_names.split(","):
        inventories.append(next((item for item in stock if item["item"] == x), None))

    print("Function:\n returns " + str(inventories) + "\n")
    return json.dumps(inventories)


# メール送信関数
def send_mail(arguments):
    args = json.loads(arguments)
    print("Function:\nreturns ")
    print({"status": "success"})
    print(
        """
mail sent as follows
=====
{}さま
いつもお世話になっております。
商品名：{}
{}
よろしくお願いします。

""".format(
            args["supplier_name"], args["items"], args["message_body"]
        )
    )
    return json.dumps({"status": "success"})


# 呼び出し可能な関数の定義
functions = [
    # 在庫チェック関数の定義
    {
        # 関数名
        "name": "inventory_search",
        # 関数の説明
        "description": "Search for inventory items. items must be separated by comma.",
        # 関数の引数の定義
        "parameters": {
            "type": "object",
            "properties": {
                "inventory_names": {
                    "type": "string",
                    "description": "Input query",
                },
            },
            # 必須引数の定義
            "required": ["input"],
        },
    },
    # メール送信関数の定義
    {
        # 関数名
        "name": "send_mail",
        # 関数の説明
        "description": "Send mail to supplier. Be sure that this function can send one mail at a time.",
        # 関数の引数の定義
        "parameters": {
            "type": "object",
            "properties": {
                "supplier_name": {
                    "type": "string",
                    "description": "suppliyer of the item",
                },
                "message_body": {
                    "type": "string",
                    "description": "message body to supplier",
                },
                "items": {
                    "type": "string",
                    "description": "an item to notify to supplier. Be sure that only one item can be notified at a time.",
                },
            },
            # 必須引数の定義
            "required": ["item_shortage"],
        },
    },
]


# JSONの16進数表現を日本語の文字列に変換する
def prettify_json(message_json):
    return json.dumps(message_json, ensure_ascii=False, indent=4)


# ユーザープロンプト
messages = [{"role": "user", "content": query}]

# AIの返答にFunction callがある限り繰り返す
while True:
    # AIへ問い合わせ
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        # 会話の履歴を与える
        messages=messages,
        # 関数の定義も毎回与える
        functions=functions,
        function_call="auto",
        temperature=0.0,
    )

    # AIの返答を取得
    message = response.choices[0]["message"]
    print("AI response: ")
    print(prettify_json(message))
    print()

    # 関数の呼び出しが必要なければループから抜ける
    if not message.get("function_call"):
        break

    # 会話履歴に追加する
    messages.append(message)

    f_call = message["function_call"]
    print("Function call: ", f_call)
    # 関数の呼び出し、レスポンスの取得
    print("Function call: " + f_call["name"] + "()\nParams: " + f_call["arguments"] + "\n")
    function_response = globals()[f_call["name"]](f_call["arguments"])

    # messagesに関数のレスポンスを追加
    messages.append(
        {
            "role": "function",
            "name": f_call["name"],
            "content": function_response,
        }
    )

# これ以上Functionを呼び出す必要がなくなった
print("Chain finished!")
print()
print("Result:")
print(message["content"])
print("message: ", messages)
