import os
import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import vectordb

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

SLACK_SIGNING_SECRET = "<YOUR_SLACK_SIGNING_SECRET>"
SLACK_BOT_TOKEN = "<YOUR_SLACK_BOT_TOKEN>"
SLACK_APP_TOKEN = "<YOUR_SLACK_APP_TOKEN>"

app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET
)

faq_db = vectordb.load("prompt-faq.csv")


@app.event("message")
def message_handler(message, say):
    # 질문에 대해 벡터 서치로 답변을 찾아서 context에 저장
    vector = vectordb.get_embedding(message['text'])
    result = vectordb.search(vector, faq_db)

    context = "--- CONTEXT ---\n"
    for i in range(3):
        context += f"질문: {result[i]['question']}\n답변: {result[i]['answer']}\n\n"

    print([
        {"role": "user", "content": context},
        {"role": "user", "content": "CONTEXT를 기반으로 다음 질문에 답변해.\n" +
         message['text']}
    ])

    # context에 담긴 질문과 답변을 기반으로 대화 생성
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": context},
            {"role": "user", "content": "CONTEXT를 기반으로 다음 질문에 답변해.\n" +
                message['text']}
        ],
        temperature=0.7,
        max_tokens=256,
    )

    say(completion.choices[0].message.content)


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
