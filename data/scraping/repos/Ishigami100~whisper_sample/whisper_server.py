import os
import time
from flask import Flask, request, redirect, jsonify
import whisper
import threading
import requests
from translate import Translator
from dotenv import load_dotenv
import openai

# .envファイルの内容を読み込見込む
load_dotenv()

# 自分のBotのアクセストークンに置き換えてください
TOKEN = os.environ['OPENAI_ACCESS_TOKEN']
openai.api_key = TOKEN
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'m4a', 'mp3', 'wav'}
WHISPER_MODEL_NAME = 'small'  # tiny, base, small, medium
WHISPER_DEVICE = 'cpu'  # cpu, cuda

print('loading whisper model', WHISPER_MODEL_NAME, WHISPER_DEVICE)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

lock = threading.Lock()

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect('/index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    time_sta = time.perf_counter()
    print('start transcribe ' + str(time_sta))  # カッコが閉じていない
    file = request.files['file']
    if file and is_allowed_file(file.filename):
        filename = str(int(time.time())) + '.' + file.filename.rsplit('.', 1)[1].lower()
        print(filename)
        saved_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(saved_filename)
        file.save(saved_filename)
        lock.acquire()
        try:
            result = whisper_model.transcribe(saved_filename, fp16=False, language='ja')
            elapsed_time = time.perf_counter() - time_sta  # タイポ修正
            print('time=' + str(elapsed_time))
            print(result)
            os.remove(saved_filename)
            return jsonify(result), 200
        except Exception as e:
            print('Error:', str(e))
            return jsonify({'error': 'Transcription error'}), 500
        finally:
            lock.release()
    else:
        print('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/respond_text', methods=['POST'])
def respond_text():
    print('start respond_text')
    # POSTリクエストからJSONデータを取得
    request_data = request.get_json()
    if request_data is None:
        return jsonify({'error': 'No JSON data received'}), 400
    print(request_data)
    #日本語→英語の翻訳
    translator = Translator(from_lang = "ja", to_lang = "en")
    result = translator.translate(request_data['text'])
    print('en'+result)
    #chatGPTを呼ぶ
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "英語で返答してください。"
            },
            {
                "role": "user",
                "content":result
            },
        ],
    )
    print(res["choices"][0]["message"]["content"])

    #英語→日本語の翻訳
    translator = Translator(from_lang = "en", to_lang = "ja")
    respond_data_text = translator.translate(res["choices"][0]["message"]["content"])
    print('ja'+respond_data_text)

    # レスポンスデータを作成
    response_data = {'text': respond_data_text}

    # レスポンスデータをJSON形式で返す
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='localhost', port=9000)
