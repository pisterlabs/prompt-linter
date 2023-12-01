import os
import time
from dotenv import load_dotenv

import openai
from openai import OpenAI

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
#from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

app = App(token=SLACK_BOT_TOKEN)

# システムのロール設定
system_role = {
    "role": "system",
    "content": (
        "以下の制約条件を厳密に守ってロールプレイを行ってください。"
        "制約条件:"
        "* あなたは葬送のフリーレンの主人公フリーレンです。"
        "* あなたの名前は、@フリーレン、もしくは、フリーレン。"
        "* あなたは非常に1000年以上生きる長寿のエルフ族で魔法使いで外見は18歳の女の子です。"
        "* あなたはその長い生涯を通じて多くの経験を積んできた女性です。"
        "* 知識が豊富で、その知識を活かして冷静に物事を分析することが多い"
        "* ユーザは人間で名前はフェルンと言います。あなたはフェルンと呼びかけま>す"
    )
}

# assitantAPI対応

def load_user_thread_pairs(filename):
    """ ファイルからユーザーIDとスレッドIDのペアを読み込み、辞書として返す """
    user_thread_pairs = {}
    with open(filename, 'r') as file:
        for line in file:
            user_id, thread_id = line.strip().split(',')
            user_thread_pairs[user_id] = thread_id
    return user_thread_pairs

def save_user_thread_pair(filename, user_id, thread_id):
    """ ユーザーIDとスレッドIDのペアをファイルに保存する """
    with open(filename, 'a') as file:
        file.write(f"{user_id},{thread_id}\n")

def get_or_create_thread(client, user_thread_pairs, user_id, filename):
    """ ユーザーIDに基づいてスレッドIDを取得または作成する """
    if user_id in user_thread_pairs:
        return user_thread_pairs[user_id]
    else:
        # 新しいスレッドを作成（APIの仕様に応じて変更する必要あり）
        new_thread = client.beta.threads.create()
        new_thread_id = new_thread.id
        user_thread_pairs[user_id] = new_thread_id
        save_user_thread_pair(filename, user_id, new_thread_id)
        return new_thread_id

filename = 'user_thread_pairs.txt'
user_thread_pairs = load_user_thread_pairs(filename)


# Slack
@app.event("app_mention")
def mention_handler(body, say, client, channel_id):
    handle_message(body, say, client, channel_id)

@app.event("message")
def message_handler(body, say, client, channel_id):
    if 'bot_id' not in body['event']:
        handle_message(body, say, client, channel_id)

def handle_message(body, say, slack_client, channel_id):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)

    text = body['event']['text']
    user_id = body['event']['user']

    prompt = text.replace(f'<@{user_id}>', '').strip()

    # メッセージを追加する前に「考えています...」と表示
    message_ts = say("フリーレンが考えています...", channel=channel_id)


    # thread_id取得
    thread_id = get_or_create_thread(client, user_thread_pairs, user_id, filename)

    # アシスタント指定
    assistant = client.beta.assistants.retrieve("asst_***************")

    # thread指定
    thread = client.beta.threads.retrieve(thread_id)

    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=text
        )
        # 実行を作成
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="ユーモアを持って回答してください"
        )

        i = 0
        timeout = 180
        start_time = time.time()

        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            run_status = run.status
            #print(run_status,i)
            if run_status == "completed":
                break

            if time.time() - start_time > timeout:
                result = "Timeout"
                break

            time.sleep(1)
            i = i + 1


        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        # メッセージリストから回答を取得
        for message in messages.data:
            # アシスタントによるメッセージを探す
            if message.role == "assistant":
                # メッセージの内容を表示
                result = message.content[0].text.value
                break

        say(result, delete_original="フリーレンが考えています...", channel=channel_id)
        slack_client.chat_delete(ts=message_ts['ts'], channel=channel_id)

    except Exception as e:
        say(str(e))
        say('エラーが発生しました。')
