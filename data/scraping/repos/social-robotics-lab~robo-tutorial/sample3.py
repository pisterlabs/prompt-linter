from robottools import RobotTools
import openai
import speech_recognition as sr
import time

rt = RobotTools('192.168.11.41', 22222)

# OpenAI APIキーをセットアップ
openai.api_key = 'APIキーを入力'

# OpenAIのGPTに接続
model_engine = 'gpt-3.5-turbo'

# 音声認識器を作成
r = sr.Recognizer()

# ChatGPTへのリクエストに含めるパラメータ
params = [
    {'role': 'system', 'content': 'あなたはユーザーの雑談相手です。'}
]

# マイクから音声を連続認識
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)  # ノイズ対策（オプション）
    while True:
        # print('何か話してください...')
        
        audio = r.listen(source)
        
        try:
            user_input = r.recognize_google(audio, language='ja-JP')
            print('USER: ' + user_input)
        except sr.UnknownValueError:
            # print('Google Speech Recognition could not understand audio')
            continue
        except sr.RequestError as e:
            # print('Could not request results from Google Speech Recognition service; {0}'.format(e))
            continue

        # ユーザーの入力を送信し、返答を取得
        params.append({'role': 'user', 'content': user_input})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=params
        )
        message = response.choices[0].message.content
        params.append({'role': 'assistant', 'content': message})
        print('ROBOT: ' + message)

        # ロボットに返答を発話させる
        d = rt.say_text(message)
        m = rt.make_beat_motion(d, speed=1.5)
        rt.play_motion(m)

        # 発話中は音声認識を止める
        time.sleep(d)