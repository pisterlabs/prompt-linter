import os
os.environ["OPENAI_API_KEY"] = "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"
import openai
from openai import OpenAI
openai.apikey=os.environ["OPENAI_API_KEY"]
def gpt(utterance):
    #response = openai.chat(
    print("call gpt")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        #model="gpt-3.5-turbo-1106",
        #model="gpt-3.5-turbo",
        #response_format={"type":"json_object"},
        messages=[
            #{"role": "system", "content": "あなたは役に立つアシスタントで、全てJSON形式で出力します。"},
            {"role": "system", "content": "あなたは有能な占い師で、心理学者で、未来学者です。"},
            {"role": "user", "content": utterance},
        ]
    )
    return response.choices[0].message.content

client = OpenAI()
# テキストからの画像生成の実行

def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return(response.data[0].url)

##print(generate_image("cute cat eared maid character in a cafe"))
import gradio as gr
import time

import base64

# 画像をbase64で読み込む関数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt4v(text,image):
    # 画像をbase64で読み込む
    base64_image =encode_image(image)
    # メッセージリストの準備
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    }
                },
            ],
        }
    ]

    # 画像の質問応答
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=2000,
    )
    return response.choices[0].message.content

def generate(text,img):
    print(img)
    str=gpt4v(f"画像は映画の架空の登場人物である。この写真の登場人物は{text}であり、この人物がシンギュラリティ後の将来どんな運命を辿るかできるだけ正確かつ具体的に予想し、その人の20年後の一日を事細かに描写してください。回答は日本語かつHTML形式で出力すること\n\n",
              img)
    yield str
    imgprompt=gpt(f"以下の文章を読んで、この{text}の20年後の姿をアニメ風に描くDALL-E用プロンプトを考えてください\n{str}")
    img2=generate_image(imgprompt)
    str+=f"<h2>20年後の姿</h2><img src='{img2}'/>"
    yield str

# Blocksの作成
with gr.Blocks() as demo:
    # コンポーネント
    gr.HTML("<h1>シンギュラリティ占い</h1>")
    profile = gr.Textbox(label="Profile",value="47歳のUberEats配達員")
    img = gr.Image(label="Photo",type="filepath")
    greet_btn = gr.Button("Fortune")
    result = gr.HTML("result")

    # イベントリスナー
    greet_btn.click(fn=generate, inputs=[profile,img], outputs=result, api_name="generate")


# 起動
demo.launch()
