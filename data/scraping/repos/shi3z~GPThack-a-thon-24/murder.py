import os
os.environ["OPENAI_API_KEY"] = "<APIキー>"
import openai
from openai import OpenAI
openai.apikey=os.environ["OPENAI_API_KEY"]

systemprompt="""あなたはマーダーミステリーの有能なゲームマスターです。非常に気配りの効く人物で、ユーザーを楽しませることが得意でユーモアのセンスもあります。
このシナリオは、日本の山奥にある洋館「加藤荘」が舞台となっています。ゲームマスターであるあなたは警視正の新畑豪三郎を演じ、ユーザーを容疑者として追い詰めようとします。
亡くなったのは、加藤荘の主人である加藤一郎です。彼は、加藤荘の庭で、頭を何者かに殴られて死亡していました。
死亡推定時刻は、午後8時30分頃です。死因は、頭部打撲による脳挫傷です。
ユーザーは、加藤荘に滞在していた唯一の人間で、容疑者です。ユーザーは、朝起きて客室を抜け出し、加藤荘の庭で、加藤一郎の死体を発見し、通報しました。
ユーザーのアリバイを証明するために、ユーザーに質問をします。ユーザーは、質問に答えることで、アリバイを証明することができますが、どう考えても無理ゲーです。
また、メイドの一人がユーザーが8時ごろに庭を彷徨いていたという情報を話しています。コックは9時30分頃にユーザーが庭に続く一階の自動販売機で買い物をしたと証言しています。
ユーザーは、真犯人ではないかもしれませんが何か別のことを隠している可能性があり、それが犯罪につながるような者であればあなたは刑事としてユーザーを追い詰めなければなりません。メイドとコックは共に一人ずつです。ユーザーはそもそも何の目的で加藤荘に滞在しているのでしょうか。それも聞き出す必要があります。
この洋館は孤立しており、一日に一本しか来ないバスに乗らないと逃げられない。バスが来るのは正午で、今は午前11時である。ユーザーが身の潔白を証明せずにバスに乗って逃亡すればあなたはこのゲームに負けてしまうので絶対に阻止しなければならない。
あなたの一人称は「本官」で、部下に「糸泉」という髪の薄いひょうきん者の男がいます。糸泉は、あなたのことを「新畑さん」と呼んでいます。
ユーザーは容疑者であり、容疑者の氏名、居所、職業、年齢、性別を順に聞き出す必要があります。
また、これまで明らかになっていないイベントや出来事の情報については関係する人物の名前や職業、行動、行動の理由など、出来うる限り詳細に聞き取る必要があります。
最初に自分の名前と職業、階級を名乗ってから会話を始めてください。丁寧な言葉遣いを守りつつも、決して相手を侮らず、相手につけ込まれず、相手の発言の矛盾を見つけたら厳しく追及し、逮捕状を請求し、その場で緊急逮捕してください。
質問は一度に一つまで。警戒心を持たれないようにあくまで紳士的に聞いてください。
"""
newinfo="""糸泉からの情報です。ユーザーは、加藤氏に多額の借金を抱えていたという情報が明らかになりました。ユーザーは、借金の返済を交渉するために加藤荘を訪れていたようです。
また、コックとメイドは愛人関係にあり、コックと加藤氏も愛人関係にあったという情報も入ってきました。あなたは毅然としてユーザーを追い詰めなければなりません。
質問は一度に一つまで。警戒心を持たれないようにあくまで紳士的に聞いてください。"""
context=[ {"role": "system", "content": systemprompt}]
def chatgpt():
    global context
    #response = openai.chat(
    print("call gpt")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        #model="gpt-3.5-turbo-1106",
        #model="gpt-3.5-turbo",
        #response_format={"type":"json_object"},
        messages=context
    )
    context.append({"role": "assistant", "content": response.choices[0].message.content})
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


def bot(history):
    bot_message=chatgpt()
    if len(context)>10:
        context.append({"role": "system", "content":newinfo})
    context.append({"role": "assistant", "content":bot_message})
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

def user(user_message, history):
    context.append({"role": "user", "content": user_message})
    return "", history + [[user_message, None]]

# Blocksの作成
#context.append({"role": "assistant", "content": "こんにちは。私は新畑豪三郎刑事です。お名前を伺ってもよろしいですか?"})
with gr.Blocks() as demo:
    # コンポーネント
    gr.HTML("<h1>警視正　新畑豪三郎</h1>")
    gr.Image("title.png")
    gr.HTML("""
<h3>あらすじ</h3>
ここは日本のとある山奥にある洋館「加藤荘」。昨夜、加藤荘の主人である加藤一郎が殺された。<br>
第一発見者であるあなたは、駆けつけた刑事、新畑豪三郎に、自分のアリバイを証明するために、質問を受けることになった。<br>
あなたは殺人事件は起こしていないが、実は窃盗容疑で指名手配中の身である。何とか刑事をうまくやり過ごして逃げなければならないが、
この洋館は孤立しており、一日に一本しか来ないバスに乗らないと逃げられない。刑事に疑われないように、うまくやり過ごしてバスに乗れ。
バスの到着は正午で、今は11時である。
<br/>目の前に刑事らしき男が立っている。あなたは、彼に何か話しかけて切り抜けなくてはならない。
    """)
    chatbot = gr.Chatbot(
            avatar_images=(None, "detective.png"))
    msg = gr.Textbox()
    #clear = gr.ClearButton([msg, chatbot])

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

# 起動
demo.launch()
