import os
# from dotenv import load_dotenv
import openai

# APIキーの設定
# load_dotenv()
# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.getenv('OPENAI_API_KEY')

def analyze_emotion(text):

    # OpenAI APIを使用して感情分析を実行する
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "あなたは感情分析サービスです。userから送信された文章の感情がポジティブなら「9」ネガティブなら「0」どちらでもないなら「4」を基準としスコアリングしてください"},
        {"role": "user", "content": "今日はとっても素敵な一日だった"},
        {"role": "assistant", "content": "7"},
        {"role": "user", "content": text}
    ]
    )


    # APIのレスポンスから感情分析結果を取得する
    result = response["choices"][0]["message"]["content"]
    print(result)
    return result