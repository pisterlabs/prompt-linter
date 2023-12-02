import openai
import json
import os
from firebase_functions import https_fn


def main(req: https_fn.Request) -> https_fn.Response:

    openai.api_key = os.environ.get("OPENAI_API_KEY")

    count = 10
    context = """
    商品名: Full Cover Bikini 
    商品カテゴリー: 男性用下着
    商品の特徴:素肌に直接触れる下着のために独自に開発した素材「nova wool® melty plus」を使用。汗蒸れ・汗冷え・汗臭を解消する消臭・抗菌機能に加え、素肌を清潔かつ快適に保つ調温・調湿機能に長けています。素肌へのストレスをクリアにし、第二の肌となってあなたの活動を支えます。食い込まないを追求した設計がストレスゼロな着用感を実現。
    キャンペーン内容: 夏にぴったりの下着の紹介
    """
    example_json = """
    [
      {"text": "Everybody Hurts"},
      {"text": "Nothing Compares 2 U"},
      {"text": "Tears in Heaven"},
      {"text": "Hurt"},
      {"text": "Yesterday"}
    ]
    """
    
    messages=[
      {"role": "system", "content": """
       あなたは優秀なコピーライターです。
       あなたはユーザーから、コピーを作成するためのいくつかの情報受け取ります。
       その情報からインスタグラムの広告で使えるようなキャッチーなコピーを作成してください。
       JSONの配列形式で返却してください。
       """},
      {"role": "assistant", "content": example_json}, 
      {"role": "user", "content": f"""
       以下の情報を参考にして、
       コピーを {count}個作成してください。

       参考情報:{context}

       """}
    ]
    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=400
    )   
    print(response)

    # return response["choices"][0]["message"]["content"]
    # return https_fn.Response(response["choices"][0]["message"]["content"])
    return json.dumps({"data" : response["choices"][0]["message"]["content"]})


