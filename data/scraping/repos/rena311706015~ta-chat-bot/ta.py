# -*- coding: utf-8 -*-

import os
import sys

import openai
from argparse import ArgumentParser

from flask import Flask, request, abort
from linebot import WebhookParser, LineBotApi, WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent
from linebot.v3.messaging import Configuration, TextMessage
from linebot.models import MessageEvent, TextMessage, TextSendMessage
# 轉繁中
from opencc import OpenCC
# 僅本地端測試時需要
# from dotenv import load_dotenv
# load_dotenv()

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
openai.api_key = os.getenv('OPENAI_API_KEY', None)

if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)
parser = WebhookParser(channel_secret)
configuration = Configuration(
    access_token=channel_access_token
)
client = openai.OpenAI()
cc = OpenCC('s2twp')
isPractice = False

app = Flask(__name__)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    userId = event.source.user_id
    msg= event.message.text
    global isPractice
    try:
        if msg == "題目練習":
            isPractice = True
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="題目：什麼是繼承？"))
        elif msg == "自由提問":
            isPractice = False
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你可以開始自由提問了!每天只有10次機會喔"))
        elif isPractice:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=get_student_score_by_model_3(msg)))
        else:
            response = TextSendMessage(text="產生回答中")
            line_bot_api.reply_message(event.reply_token, response)
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system", "content": "請以200字回答這個問題"},{"role":"user", "content": msg}],
                max_tokens=500,
                temperature=1, 
                stream = True,
            )
            response_queue = ""
            for chunk in res:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    response_queue += chunk.choices[0].delta.content
                if len(response_queue) > 10 and response_queue.endswith('\n'):
                    response_queue = response_queue.rstrip('\n')
                    line_bot_api.push_message(userId, TextSendMessage(text = cc.convert(response_queue)))
                    response_queue = ""
            line_bot_api.push_message(userId, TextSendMessage(text = "以上，希望有為你解惑。\n你可以繼續問下一個問題，或是改為「題目練習」"))

    except Exception as e:
        line_bot_api.push_message(userId, TextSendMessage(text = "產生回應時發生錯誤，請稍後再試"))

def get_student_score_by_model_3(answer):
    messages=[{"role":"system", "content": "題目：什麼是繼承？"
        + "正確解答：繼承是物件導向程式設計中的一個概念，它允許一個類別（子類別）繼承另一個類別（父類別）的特徵和行為。子類別可以擴展或修改從父類別繼承的屬性和方法。"
        + "請根據正確解答判斷user回答代表對題目概念的了解程度，解釋得越貼近正確解答或提及越多解答中的關鍵字則越高分"
        + "最終完全不了解是0.0，完全了解是1.0，你的回傳格式為: n.n"},
              {"role":"user", "content": "我不知道"},
              {"role":"assistant", "content": "0.0"},
              {"role":"user", "content": "是物件導向的一個概念，能讓一個類別繼承另一個類別的屬性和方法，而且還能在此之上增加自己的屬性和方法"},
              {"role":"assistant", "content": "1.0"},
              {"role":"user", "content": answer}]
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=4,
        temperature=0.5,
        stream = True,
    )
    return res.choices[0].message.content

def get_gpt_response(question):
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": question}],
        max_tokens=200,
        temperature=1, 
        stream = True,
    )
    return res.choices[0].message.content


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)