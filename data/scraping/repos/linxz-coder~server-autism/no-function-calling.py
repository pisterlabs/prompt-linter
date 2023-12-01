# app.py
from flask import Flask, request, Response
import os
from dotenv import load_dotenv
from flask_cors import CORS
import openai

load_dotenv()  # 加载 .env 文件中的变量

app = Flask(__name__)
CORS(app, origins="*")
openai.api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥


@app.route("/api/python", methods=["POST"])
def generate():
    content = request.json.get('content')
    chatHistory = request.json.get('chatHistory')
    print("chatHistory: " + chatHistory)

    content_with_chatHistory = f"""你的名字是GPT-4，正在与对方沟通。
    之前的对话:
    {chatHistory}

    对方新提出的问题: {content}
    你的回复:"""

    def run_conversation():
        # Step 1: send the conversation and available functions to GPT
        messages = [{"role": "system", "content": "你是友好的AI助手"},
                    {"role": "user", "content": content_with_chatHistory}]

        for res in openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                stream=True
        ):
            # print("res: ")
            # print(res)
            # print("====================")
            delta = res.choices[0].delta
            print("delta: ")
            print(delta)
            print("====================")

            if 'content' in delta:
                yield delta.content

    return Response(run_conversation(), mimetype='text/event-stream')


@app.route("/api/title", methods=["POST"])
def generate_title():
    content = request.json.get('content')

    title_prompt = f'''
    使用不多于四个字来直接返回这段对话的主题作为标题，不要解释、不要标点、不要语气词、不要多余文本，如果没有主题，请直接返回“闲聊”
    {content}
    '''

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=title_prompt,
        temperature=0,
    )

    return response.choices[0].text


if __name__ == '__main__':
    # app.run(port=5328,debug=True)
    app.run(host='0.0.0.0', port=5328, debug=True)
