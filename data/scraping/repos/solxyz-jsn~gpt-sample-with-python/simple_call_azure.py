# -*- coding: utf-8 -*-

"""リクエストサンプル

OpenAI APIを使用して、単純なリクエストを送るサンプル

"""

import openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
import sys
import logging


def get_completion(prompt: str, engine: str) -> str:
    """
    プロンプトを受け取り、OpenAIにリクエストを送信して、結果を返す

    Parameters
    ----------
    prompt : str
        プロンプト
    model : str, optional
        エンジンの指定, by default "gpt-3.5-turbo"
    Returns
    -------
    str
        OpenAIからのレスポンス
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        # エンジンの指定
        engine=engine,
        deployment_id=engine,
        # ユーザーの発言として、日本語を入力
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0]["message"]["content"]


# .envファイルから環境変数を読み込み
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
# APIキーを環境変数から取得
openai.api_key: str = token.token
# openai.api_key: str = os.getenv("OPENAI_API_KEY")
openai.api_base: str = os.getenv("OPENAI_API_BASE")
openai.api_version: str = os.getenv("OPENAI_API_VERSION")
openai.api_type: str = os.getenv("OPENAI_API_TYPE")
ai_model: str = os.getenv("AZURE_MODEL")

# # 翻訳する文字列
# word: str = "Hello World"
# # バックティックを使うことで、入力文字列をプロンプトから分離できます。
# # これはプロンプトインジェクションを防ぐために推奨されています。
# # 参照:<https://ja.wikipedia.org/wiki/%E3%83%97%E3%83%AD%E3%83%B3%E3%83%97%E3%83%88%E3%82%A8%E3%83%B3%E3%82%B8%E3%83%8B%E3%82%A2%E3%83%AA%E3%83%B3%E3%82%B0#%E6%82%AA%E6%84%8F%E3%81%AE%E3%81%82%E3%82%8B%E8%A1%8C%E7%82%BA>
# prompt: str = f"""日本語に翻訳してください: ```{word}```"""
# # 翻訳された文字列が出力される
# print(get_completion(prompt=prompt, engine=ai_model))

# # CSVを定義
# csv: str = f"""year_month,detail \n
# 1981(昭和56)年2月,受託ソフトウェア開発事業を目的に埼玉県大宮市にて株式会社エポックシステム設立 \n
# 1982(昭和57)年2月,東京都港区新橋に本社を移転 \n
# 1984(昭和59)年12月,東京都港区芝大門に本社を移転 """

# # プロンプトを定義
# # JSON形式を明示的に指定し、整形するように指示
# prompt: str = f"""CSVデータをJSONに整形してください。year_monthは西暦、和暦、月に分解してください（例：1981(昭和56)年2月はad:1981,japanese:昭和56,month:2）。: ```{csv}```"""

# print(get_completion(prompt=prompt, engine=ai_model))

# # 一連の命令が含まれるテキストを定義
# text: str = f"""Pythonを学ぶには、無料のオンラインチュートリアルや本で基本的な概念を学びます。\
#     続いてSOLXYZ Academyなどの学習プラットフォームのコースを受講します。\
#     受講後はシンプルなプロジェクトを作ってコードを書きます。\
#     またエラーを解決するデバッグ技術を学びます。\
#     さらなる深耕のためにPythonコミュニティに参加します。"""

# # 成型するフォーマットを含むプロンプトを定義
# prompt: str = f"""与えられるテキストに、一連の命令が含まれる場合は、次のように成型してください。:

#     ステップ１：...
#     ステップ２：...
#     ...
#     ステップn：...

#     もし、テキストに命令が含まれない場合は、「命令が含まれていません」と出力してください。

#     テキスト：```{text}```"""

# # 指定した箇条書きで出力される
# print(get_completion(prompt=prompt, engine=ai_model))

# Chain of Thought
prompt: str = f"""私は市場に行って10個のリンゴを買いました。隣人に2つ、修理工に2つ渡しました。それから5つのリンゴを買って1つ食べました。残りは何個ですか？ステップバイステップで答えてください。"""
# 翻訳された文字列が出力される
print(get_completion(prompt=prompt, engine=ai_model))
