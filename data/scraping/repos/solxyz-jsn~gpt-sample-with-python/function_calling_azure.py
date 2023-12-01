# -*- coding: utf-8 -*-

"""OpenAI APIを使って関数を呼び出す

OpenAI APIを使って関数を呼び出すサンプルです。
次のサイトの記事を参考にしています。
https://dev.classmethod.jp/articles/function-calling-blog-search-and-answer/

"""

import openai
import os
import sys
import urllib.request
import streamlit as st
import logging
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# .envファイルから環境変数を読み込み
load_dotenv()

# ログの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
# ChatGPT-3.5のモデルのインスタンスの作成
openai.api_key: str = token.token
# APIキーを環境変数から取得
openai.api_base: str = os.getenv("OPENAI_API_BASE")
openai.api_version: str = os.getenv("OPENAI_API_VERSION")
openai.api_type: str = os.getenv("OPENAI_API_TYPE")
ai_model: str = os.getenv("AZURE_MODEL_16K")


def get_blog_contents(url: str) -> str:
    """
    指定したURLのブログ記事の内容を取得する
    Args:
        url (str): ブログ記事のURL
    Returns:
        str: ブログ記事の内容
    """
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as res:
        body = res.read()
    html_doc = body.decode()
    soup = BeautifulSoup(html_doc, "html.parser")
    contents = soup.find("div", class_="content")
    texts = [c.get_text() for c in contents.find_all("p")]
    texts = "\n\n".join(texts)

    return texts[:4000]


def search_blog(query_str: str) -> str:
    """
    指定したキーワードでソルクシーズ公認ブログを検索して、URLのリストを得る。
    Args:
        query_str (str): 検索キーワード
    Returns:
        str: URLのリスト
    """
    search = GoogleSearch(
        {
            "q": f"site:solxyz-blog.info {query_str}",
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }
    )

    result = search.get_dict()

    address_list = [result["link"] for result in result["organic_results"]]
    return str(address_list)


# モデルの指定
# model_name = "gpt-3.5-turbo-16k-0613"
# fuction定義のリスト
functions = [
    {
        "name": "search_blog",
        "description": "指定したキーワードでソルクシーズ公認ブログを検索して、URLのリストを得る。",
        "parameters": {
            "type": "object",
            "properties": {
                "query_str": {
                    "type": "string",
                    "description": "検索キーワード",
                },
            },
            "required": ["query_str"],
        },
    },
    {
        "name": "get_blog_contents",
        "description": "指定したURLについてその内容を取得して、パースした結果のテキストを得る。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "内容を取得したいページのURL",
                },
            },
            "required": ["url"],
        },
    },
]


# タイトルの作成
st.title("🐟Function Calling🐟")
# 入力フォームの作成
query_str = st.text_input("検索ワードを入力してください。")
# 質問が入力された時、OpenAIのAPIを実行
if query_str:
    # 質問の作成
    question = f"""
    「```{query_str}```」について、まずソルクシーズ公認ブログを検索した結果のその上位3件を取得します。
    その後、それぞれのURLについてその内容を取得して、パースした結果のテキスト得ます。
    そしてそれらのパースした結果をまとめ、最終的な答えを１つ生成してください。
    """
    # 最大のリクエスト回数
    MAX_REQUEST_COUNT = 10
    # メッセージの履歴を初期化
    message_history = []
    for request_count in range(MAX_REQUEST_COUNT):
        function_call_mode = "auto"
        if request_count == MAX_REQUEST_COUNT - 1:
            function_call_mode = "none"
        response = openai.ChatCompletion.create(
            # エンジンの指定
            engine=ai_model,
            deployment_id=ai_model,
            messages=[
                {"role": "user", "content": question},
                *message_history,
            ],
            functions=functions,
            function_call=function_call_mode,
        )
        # messageがfunction_callを含む場合
        if response["choices"][0]["message"].get("function_call"):
            message = response["choices"][0]["message"]
            if message.get("content") is None:
                message.content = " "  # 一旦空白で設定する。AzureのAPIではcontentが設定されていないとエラーになるため。
            message_history.append(message)
            function_call = response["choices"][0]["message"].get("function_call")
            function_name = function_call.get("name")
            function_arguments = function_call.get("arguments")
            if function_name in [f["name"] for f in functions]:
                function_response = eval(function_name)(**eval(function_arguments))
            else:
                raise Exception
            message = {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
            message_history.append(message)
        # messageがfunction_callを含まない場合は回答を表示
        else:
            st.write(response.choices[0]["message"]["content"].strip())
            break
