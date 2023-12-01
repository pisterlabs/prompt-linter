from openai import OpenAI
from text_to_speech import text_to_speech
from speech_to_text import speech_to_text

client = OpenAI()

# テンプレートの準備
template = "あなたは音声対話型チャットボットです。完結で短い文章で回答してください。"

# メッセージの初期化
messages = [{"role": "system", "content": template}]

# 区切り文字の定義
punctuations = ["。", "？", "！"]

# ユーザーからのメッセージを受け取り、それに対する応答を生成
while True:
    # 音声をテキストに変換
    user_message = speech_to_text()

    # テキストが空の場合は処理をスキップ
    if user_message == "":
        continue

    print("あなたのメッセージ: \n{}".format(user_message))
    messages.append({"role": "user", "content": user_message})

    # ストリーミングでチャットボットの回答を生成
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, stream=True
    )

    # レスポンスは細かく分割されているので、結合してメッセージを組み立てる
    message_buffer = ""
    bot_message = ""
    print("チャットボットの回答: ", end="", flush=True)
    for chunk in response:
        finish_reason = chunk.choices[0].finish_reason
        if finish_reason == "stop":
            break
        message_buffer += chunk.choices[0].delta.content
        bot_message += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="", flush=True)
        # 最後が「。」や「？」、「！」で終わっていたら、途中でもいったん再生
        if any(message_buffer.endswith(punctuation) for punctuation in punctuations):
            text_to_speech(message_buffer)
            message_buffer = ""

    # 未再生のメッセージがあれば再生
    if message_buffer != "":
        text_to_speech(message_buffer)

    # 最後に改行するためのprint
    print()
    messages.append({"role": "assistant", "content": bot_message})
