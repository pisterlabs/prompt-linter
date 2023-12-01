import datetime
import json
import os

from openai_wrapper import OpenAIWrapper
from speech_recognizer import SpeechRecognizer
from translator import Translator
from voice_synthesizer import VoiceSynthesizer

# メディアタイプの定義
MEDIA_TYPE_JSON = 0x00
MEDIA_TYPE_AUDIO = 0x01
MEDIA_TYPE_IMAGE = 0x02


class MediaDataHandler:
    def __init__(self, config):
        self.config = config
        self.recognizer = SpeechRecognizer(config, "whisper")
        self.synthesizer = VoiceSynthesizer(config, "openai")

        self.deepl = Translator(config.deepl_api_key)
        self.openai = OpenAIWrapper(config.openai_api_key)
        self.canSpeak = True

    ##########################################################
    # 受信関数

    # JSON受信時の処理
    async def handle_json(self, json_message, websocket):
        print("Received json message:" + json_message)

    # オーディオ受信時の処理
    async def handle_audio_data(self, audio_data, websocket):
        print("Received audio data")

        received_time = datetime.datetime.now()
        filename = received_time.strftime("%Y%m%d%H%M%S") + ".wav"
        filepath = os.path.join("audio", filename)
        # フォルダがなければ作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(audio_data)


        # 音声認識
        result,lang = self.recognizer.recognize(filepath)

        if result != "":
            await websocket.send("[whisper]"+result)
            # メッセージ処理
            await self.on_receive_message(result, websocket)

        # ファイルを削除(例外発生時はログ)
        try:
            os.remove(filepath)
        except:
            print("Failed to delete file: " + filepath)


    # 画像受信時の処理
    async def handle_image_data(self, image_data, websocket):
        print("Received image data")
        await websocket.send("received image data")

    # テキスト受信時の処理
    async def handle_text_message(self, message, websocket):
        await self.on_receive_message(message, websocket)


    # メッセージ受信時の処理
    async def on_receive_message(self, message, websocket):
        print("Received message:" + message)
        # メッセージが空なら戻る
        if message == "":
            return


        received_time = datetime.datetime.now()
        result = self.openai.query(message,"gpt4test")
        # かかった時間を計算
        lapse_time = datetime.datetime.now() - received_time

        if self.canSpeak:
            self.synthesizer.synthesize(result, "audio/gpt4test.mp3")


        # gptのテキストメッセージを送信
        await self.send_message( websocket,f"({lapse_time.total_seconds():.2f}秒) : {result}")

    # テキストを音声に変換し、WebSocket経由でクライアントに送信する
    async def handle_text_to_speech(self, websocket, text):

        # OpenAI APIを使用してテキストを音声に変換
        audio_stream = self.openai.text_to_speech_stream(text)

        # 音声データをチャンクごとにクライアントに送信
        try:
            for audio_chunk in audio_stream:
                # チャンクの先頭に1バイト (0x01) を追加
                modified_chunk = b'\x01' + audio_chunk
                await websocket.send(modified_chunk)
        except Exception as e:
            print(f"Error while sending audio stream: {e}")


    ##########################################################
    # 以下、送信用の関数
    async def send_message(self, websocket, message):
        # テキストデータをクライアントに送信する
        if websocket.open:
            await websocket.send(message)

    # JSONを送信
    async def send_json(self, websocket, data):
        await self.send_media(websocket, data, MEDIA_TYPE_JSON)

    # audioを送信
    async def send_audio(self, websocket, data):
        await self.send_media(websocket, data, MEDIA_TYPE_AUDIO)

    # 画像を送信
    async def send_image(self, websocket, data):
        await self.send_media(websocket, data, MEDIA_TYPE_IMAGE)

    async def send_media(self, websocket, data, media_type):

        # websocket.openでなければ戻る
        if not websocket.open:
            return
        # media_typeが0の場合はJSONとして処理
        if media_type == 0x00:
            json_message = json.dumps(data)
            await websocket.send(json_message)
        else:
            # 例えば、音声は0x01, 画像は0x02とする
            media_header = bytearray([media_type])
            await websocket.send(media_header + data)
