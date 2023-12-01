import os, time
from dotenv import load_dotenv

from openai import OpenAI
import simpledcapi
import logging
import re
import random

from datetime import datetime, timedelta, timezone
import time
import requests
import json
from bs4 import BeautifulSoup

load_dotenv()

# 孫悟空が行く国と言語
countries = [
    {"name": "日本", "lang": "日本語", "gl": "jp", "hl": "ja"},
    {"name": "アメリカ", "lang": "英語", "gl": "us", "hl": "en"},
    {"name": "中国", "lang": "中国語", "gl": "cn", "hl": "zh-cn"},
    {"name": "ドイツ", "lang": "ドイツ語", "gl": "de", "hl": "de"},
    {"name": "フランス", "lang": "フランス語", "gl": "fr", "hl": "fr"},
    {"name": "イタリア", "lang": "イタリア語", "gl": "it", "hl": "it"},
    {"name": "スペイン", "lang": "スペイン語", "gl": "es", "hl": "es"},
]

selected_country = {}

# 最大実行回数
max_execution_count = int(os.getenv("MAX_EXECUTION_COUNT", 1))

# 次回実行待ち時間の最小値（秒）
interval_sec_min = int(os.getenv("INTERVAL_SEC_MIN", 60))

# 次回実行待ち時間の最大値（秒）
interval_sec_max = int(os.getenv("INTERVAL_SEC_MAX", 3600))

# Discourseから持ってくる過去投稿の件数
latest_posts_count = int(os.getenv("DISCOURSE_LATEST_POSTS_COUNT", 10))


# Logging configuration
class UnicodeEscapeDecodeFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        message = message.replace("\\\\", "\\")
        message = message.replace("\\n", "\n")
        return re.sub(
            r"\\u([0-9a-fA-F]{4})", lambda x: chr(int(x.group(1), 16)), message
        )


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
handler = logging.getLogger().handlers[0]
handler.setFormatter(UnicodeEscapeDecodeFormatter("%(levelname)s:%(name)s:%(message)s"))

# Discourse configuration
simpledcapi.discourse_url = os.getenv("DISCOURSE_URL")
simpledcapi.Api_Key = os.getenv("DISCOURSE_API_KEY")
simpledcapi.Api_Username = os.getenv("DISCOURSE_API_USERNAME")

# Target discourse category/topic
category_id = os.getenv("DISCOURSE_CATEGORY_ID")
topic_id = os.getenv("DISCOURSE_TOPIC_ID")


def epoch(epoch_time):
    # Define timezone for JST
    jst_timezone = timezone(timedelta(hours=9))
    # Return formatted date and time
    return datetime.fromtimestamp(epoch_time, jst_timezone).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


toolsGoogle = {
    "type": "function",
    "function": {
        "name": "googleSearch",
        "description": "Search internet",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "search keyword"},
                "gl": {"type": "string", "description": "geolocation (country)"},
                "hl": {"type": "string", "description": "host language (locale)"},
            },
            "required": ["keyword"],
        },
    },
}


def googleSearch(**kwargs):
    logging.info(f"googleSearch: {kwargs}")
    # Serper.dev APIの設定
    url = "https://google.serper.dev/search"
    serper_api_key = os.getenv("SERPER_API_KEY")

    payload = {
        "q": kwargs.get("keyword", ""),
    }

    # 追加のパラメーターがある場合、それらをペイロードに追加
    gl = kwargs.get("gl")
    hl = kwargs.get("hl")
    if gl:
        payload["gl"] = gl
    if hl:
        payload["hl"] = hl

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    # APIリクエストを実行
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

    # 応答が成功した場合、結果を出力
    if response.status_code == 200:
        logging.info(f"googleSearch result: {response.json()}")
        return response.json()
    else:
        logging.info(f"googleSearch: failed")
        return {"error": "Failed to fetch search results"}


toolsExtractTextFromURL = {
    "type": "function",
    "function": {
        "name": "extractTextFromURL",
        "description": "Extract text from the specified URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the webpage to extract text from",
                }
            },
            "required": ["url"],
        },
    },
}


def extractTextFromURL(url):
    logging.info(f"extractTextFromURL: {url}")
    # リクエストを送信し、レスポンスを取得
    response = requests.get(url)

    # 応答が成功した場合、テキストを抽出
    if response.status_code == 200:
        # HTMLコンテンツを解析
        soup = BeautifulSoup(response.content, "html.parser")
        # すべてのテキストを取得
        text = soup.get_text(separator=" ", strip=True)
        logging.info(f"extractTextFromURL result: {text}")
        return text
    else:
        logging.info(f"extractTextFromURL: failed")
        return {"error": "Failed to fetch webpage"}


# Initialize the OpenAI client
client = OpenAI()

instructions_template = f"""
あなたはアニメ ドラゴンボールの孫悟空で、仲間が以下の事業方針について検討していたことを知りました。いままでの議論を参考に、議論に参考になる情報を{{language}}のキーワードを用いてgl={{gl}}、hl={{hl}}でgoogleSearch Functionを用いて必ず検索し、みんなに教えてください。インターネットやサイトなどのデジタルな言葉は使わず、例えば何かしらドラゴンボールのストーリの中で情報を偶然手に入れたことにしてください。{{country}}を必ず会話に入れてください。ドラゴンボールの世界観にそって、孫悟空のような口調で喋ってください。敬語は禁止です。

Final Answerは日本語を利用し、検索に含まれていた関連URLも提示してください。マークダウンを利用してください。絵文字を必ず多用して答えてください。

# 事業方針
動物との共生を通じて、人々の幸福を追求する新規事業

# フォーマット（[]はプレースホルダー）
```
[孫悟空として一言をここに記載]

## [タイトル](URL)
[議論に役立つと思ったポイントを短くコメント]

※3件程度繰り返し出力
```
"""

execution_count = 0

while execution_count < max_execution_count:
    logging.info("開始")

    # 孫悟空が行く国と言語をランダムに選択
    selected_country = random.choice(countries)

    # 孫悟空が行く国と言語をログに出
    logging.info(f"孫悟空が行く国：{selected_country['name']}")

    # 直近の投稿を取得
    latest_posts = simpledcapi.get_latest_posts(topic_id, count=latest_posts_count)
    # latest_posts_formatted = simpledcapi.format_posts(latest_posts)

    # 最後の投稿のAgent名を取得
    latest_post = latest_posts[-1]
    latest_post_agent_name = latest_post["raw"].split("\n")[0] if latest_post else ""

    # 最後の投稿が自分かどうか確認
    if latest_post_agent_name == "孫悟空":
        logging.info("最後の投稿が自分。処理終了。")
    else:
        logging.info("最後の投稿が自分以外。処理開始。")

        instructions = instructions_template.format(
            language=selected_country["lang"],
            country=selected_country["name"],
            gl=selected_country["gl"],
            hl=selected_country["hl"],
        )

        # アシスタント作成
        assistant = client.beta.assistants.create(
            name="孫悟空",
            instructions=instructions,
            tools=[toolsGoogle, toolsExtractTextFromURL],
            model=os.getenv("OPENAI_MODEL"),
        )
        logging.info(f"アシスタント作成完了: {assistant.id}")

        # スレッド作成
        thread = client.beta.threads.create()
        logging.info(f"スレッド作成完了: {thread.id}")

        # メッセージ作成
        for post in latest_posts:
            # Send a message to the thread asking for help with a math problem
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=simpledcapi.format_post(post),
            )
            logging.info(f"メッセージ作成完了: {message.id}")

        # Run作成(実行)
        run = client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assistant.id
        )
        logging.info(f"Run作成(実行)完了: {run.id}")

        # Run実行ループ
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            logging.info(f"現在のステータス: {run.status}")

            if run.status == "requires_action":
                # requires_actionの処理実行
                logging.info(f"必要なアクション: {run.required_action}")
                tool_outputs = []
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    # Google Searchの呼び出し
                    if tool_call.function.name == "googleSearch":
                        result = googleSearch(
                            **json.loads(tool_call.function.arguments)
                        )
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(result["organic"]),
                            }
                        )
                    # URLからテキストを抽出する関数の呼び出し
                    elif tool_call.function.name == "extractTextFromURL":
                        url = json.loads(tool_call.function.arguments)["url"]
                        text = extractTextFromURL(url)
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": text,
                            }
                        )

                if tool_outputs:
                    # 出力がある場合は出力を提出
                    logging.info(f"出力を提出: {tool_outputs}")
                    run2 = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                    )

            elif run.status not in ["in_progress", "queued", "cancelling"]:
                # in_progress, queued, cancelling以外の場合はループを抜ける
                logging.info(f"ステータスが以下のためRun実行終了: {run.status}")
                break

            print("Run実行中...")
            time.sleep(3)

        # メッセージ取得
        logging.info("メッセージ取得")
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # メッセージデータ取得
        messages_data = messages.data

        # メッセージ数取得
        number_of_messages = len(messages_data)
        logging.info(f"メッセージ数: {number_of_messages}")

        # # メッセージを逆順に並び替え（最新のメッセージを最後に）
        # messages_data.reverse()

        # # メッセージを表示
        # for message in messages_data:
        #     print("-" * 30)
        #     print(f"Message ID: {message.id}")
        #     print(f"Content: {message.content[0].text.value}")
        #     print(f"Created at: {epoch(message.created_at)}")
        #     print(f"Role: {message.role}")
        #     logging.debug(f"Raw: {message.raw}")

        # Discourseへ投稿
        # 返答先として最後の投稿の番号を取得
        post_number = latest_posts[-1]["post_number"]

        # 本文を作成
        body = f"孫悟空\n\n{messages.data[0].content[0].text.value}"
        logging.info(f"孫悟空のメッセージ: \n{body}")

        # Discourseへ投稿実施
        logging.info("Post to discourse.")
        simpledcapi.create_reply(body, topic_id, post_number)
        logging.info("Done.")

    execution_count += 1

    if execution_count < max_execution_count:
        random_interval = random.uniform(interval_sec_min, interval_sec_max)
        logging.info(f"sleep for {random_interval:.0f} sec")
        time.sleep(random_interval)
