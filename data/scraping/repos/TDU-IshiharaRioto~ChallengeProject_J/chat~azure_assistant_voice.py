import azure.cognitiveservices.speech as speechsdk
import time
import openai
import os
from websocket_server import WebsocketServer
import threading
import signal
import sys
import json

functions = [
	{
		"name":"teach_train_time_table",
		"description": "電車の運行情報を教えてください",
		"parameters":{
			"type": "object",
                "properties": {
					"line name": {
                        "type": "string",
                        "description": "聞かれている路線"
					}
			},
			"requaired":[]
		}

	},
	
]

# Global Variables
server = None
session_active = False
messages_history = []
last_input_time = time.time()
recognized_text = ""

# SIGINTハンドラ関数
def signal_handler(sig, frame):
    global server
    print("Shutting down gracefully...")
    server.shutdown()
    sys.exit(0)

# WebSocketサーバーのコールバック関数
def new_client(client, server):
    print("New client connected")

def client_left(client, server):
    print("Client disconnected")

def message_received(client, server, message):
    print("Client said: " + message)

def setup_websocket_server():
    global server
    server = WebsocketServer(host="127.0.0.1", port=5005)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    thread = threading.Thread(target=server.run_forever)
    thread.start()

def recognized(evt):
    global recognized_text
    if evt.result.text == "":
        return
    print('「{}」'.format(evt.result.text))
    recognized_text = evt.result.text

def check_activation_phrase(text):
    activation_phrases = ["鏡よ", "鏡を"]
    return any(phrase in text for phrase in activation_phrases)

def handle_activation():
    global session_active, messages_history
    session_active = True
    messages_history = [{"role": "system", "content": "あなたはスマートミラーの中に搭載されたAIアシスタントです。"},{"role": "user", "content": "鏡よ、鏡"}]
    print("はい、なんでしょう？")
    speech_synthesizer.speak_text("はい、なんでしょう？")

def get_openai_response(text):
    global messages_history
    try:
        messages_history.append({"role": "user", "content": text})
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messages_history,
            functions=functions
        )
        s = str(response['choices'][0]['message'])
        d = {}
        d = json.loads(s)
        print(s)
        if('function_call' in d):
            if(d['function_call']['name'] == 'teach_train_time_table'):
                print('運行情報')
                server.send_message_to_all('{"type":"CALL","name":"train"}')
        if('content' in d):
            print("B")
       
            
        messages_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        #print(response['choices'][0]['message']['function_call'])
        return ""

def main_loop():
    global recognized_text, last_input_time, session_active, messages_history

    while True:
        time.sleep(1)
            
        
        if(session_active):
            print(time.time() - last_input_time)
        if session_active and (time.time() - last_input_time) > 20:
            session_active = False
            messages_history = []
            print("会話を終了しました")
            continue

        if recognized_text == "":
            continue

        session_active = True
        if session_active:
            response_text = get_openai_response(recognized_text)
            if response_text:
                print(response_text)
                speech_recognizer.stop_continuous_recognition()
                server.send_message_to_all('{"type":"TEXT","text":"' + response_text + '"}')
                speech_synthesizer.speak_text(response_text)
                speech_recognizer.start_continuous_recognition()

        elif check_activation_phrase(recognized_text):
            handle_activation()

        last_input_time = time.time()
        recognized_text = ""

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    setup_websocket_server()

    # Azure and OpenAI setup
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_version = "2023-07-01-preview"
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

    speech_key = os.getenv('AZURE_SPEECH_KEY')
    service_region, language = "japaneast", "ja-JP"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region, speech_recognition_language=language)
    audio_input_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input_config)

    audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = 'ja-JP-NanamiNeural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

    print("話しかけてください...")

    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.start_continuous_recognition()

    main_loop()
