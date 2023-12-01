# -*- coding: utf-8 -*-
import os
import openai
import speech_recognition as sr
import requests
import subprocess

openai.api_key = 'APIキーを記入'
elevenlabs_apikey = 'APIキーを記入'



# レコーダーのインスタンス化
r = sr.Microphone()

while True:

    # マイクからの音声を取得
    with r as source:
        print("話してください")
        # audio = sr.Recognizer().record(source, duration=8) #待ち時間短縮
        audio = sr.Recognizer().record(source, duration=4) #待ち時間短縮

    try:
        # GoogleのWebスピーチAPIを使用して音声をテキストに変換
        text = sr.Recognizer().recognize_google(audio, language='ja-JP')
        print("あなたが言ったこと: " + text)

        # ChatGPTにテキストを渡す
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            # model="text-davinci-002",
            max_tokens=100, #50    
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            messages=[
                {"role": "system", "content": "あなたは世界で最高のメンターです。思いやりの心をもち、含蓄があり象徴的で哲学的です。回答文の長さがmax_tokenに収まり、必ず意味が通った完結した文章にしてください。"},
                {"role": "user", "content": text}
            ]
        )      
        # ChatGPTからのレスポンスを表示
        print(response['choices'][0]['message']['content'])
                
        # 11labsを使用して音声をテキストに変換。
        CHUNK_SIZE = 1024
        url = "https://api.elevenlabs.io/v1/text-to-speech/zzzzzzzzzzzzzzzz" #ここを自分のボイスに差し替えると自分と会話してる気分になれる。超重要な部分
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_apikey
        }
        data = {
            "text": response['choices'][0]['message']['content'],
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        response = requests.post(url, json=data, headers=headers)
        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        # mp3ファイルを鳴らす
        subprocess.call("mpg321 output.mp3", shell=True)


    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))



