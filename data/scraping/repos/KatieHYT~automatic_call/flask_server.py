from flask_cors import CORS
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import openai
from typing import List, Optional
from abc import ABC, abstractmethod
import json
import base64
import os
from twilio.rest import Client
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_sockets import Sockets
import functools
import threading
import time
import tempfile
import speech_recognition as sr
import sys
import datetime
import random

sys.path.append("..")
from src.tools import TalkerX, TalkerCradle

class FlaskCallCenter:
    def __init__(self, remote_host: str, port: int, static_dir: str):
        self.app = Flask(__name__)
        CORS(self.app)
        self.sock = Sockets(self.app)
        self.remote_host = remote_host
        self.port = port
        self.static_dir = static_dir

        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        self.twilio_client = Client(account_sid, auth_token)
        self.sid2latlng = {}

        @self.app.route("/checkcall", methods=["POST"])
        def checkcall():
            post_data = request.json
            latlng = post_data['latlng']
            print(f"Checking call history...{latlng}")
            last_list = os.listdir(os.environ["LAST_CALL_DIR"])
            call_conversation=None
            if latlng:
                if latlng+'.txt' in last_list:
                    path = os.path.join(os.environ["LAST_CALL_DIR"], latlng+'.txt')
                    # Open the file in read mode ('r')
                    with open(path, 'r') as file:
                        # Read the entire file content into a variable
                        call_conversation = file.read() 
            response_data = {
                    "message": "POST request to /call was successful",
                    "call_conversation": call_conversation}

            return jsonify(response_data), 200

        @self.app.route("/call", methods=["POST"])
        def call():
            post_data = request.json
            call_to = post_data['call_to']
            latlng = post_data['latlng']

            call = self.twilio_client.calls.create(
                to=call_to,
                from_=os.environ['TWILIO_PHONE_NUMBER'],
                url=f"https://{self.remote_host}/",
            )
            self.save_use_record(call.sid)
            
            self.sid2latlng[call.sid] = latlng
            response_data = {"message": "POST request to /call was successful"}

            return jsonify(response_data), 200


        @self.app.route("/", methods=["POST"])
        def incoming_voice():
            print("---> inside imcomving_voice")
            XML_MEDIA_STREAM = """
            <Response>
              <Start>
                <Stream url="wss://{host}/streaming" />
              </Start>
              <Pause length="60"/>
              <Say>
            	  Hello KT
              </Say>
            </Response>
            """
            return XML_MEDIA_STREAM.format(host=self.remote_host)
        
        @self.sock.route("/streaming")
        def on_media_stream(ws):
            print("---> inside /    socket")
            agent_a = TalkerCradle(
                    static_dir=self.static_dir,
             )
            talker_x = TalkerX()

            thread = threading.Thread(target=self.conversation, args=(agent_a, talker_x))
            thread.start()

            while True:
                try:
                    message = ws.receive()
                except simple_websocket.ws.ConnectionClosed:
                    logging.warn("Call media stream connection lost.")
                    break
                if message is None:
                    logging.warn("Call media stream closed.")
                    break
                data = json.loads(message)

                if data["event"] == "start":
                    print("Call connected, " + str(data["start"]))
                    agent_a.phone_operator = self.twilio_client.calls(data["start"]["callSid"])

                elif data["event"] == "media":
                    media = data["media"]
                    chunk = base64.b64decode(media["payload"])
                    if talker_x.stream is not None:
                        talker_x.write_audio_data_to_stream(chunk)
                        
                elif data["event"] == "stop":
                    print("Call media stream ended.")
                    break

        @self.app.route("/audio/<key>")
        def audio(key):
            return send_from_directory(self.static_dir, str(int(key)) + ".mp3")

    def save_use_record(self, file_content, save_dir='./use_record'):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{formatted_datetime}.txt")
        with open(save_path, "w") as file:
            file.write(file_content)

    def reply(self, phone_operator, audio_key, duration):
        phone_operator.update(
            twiml=f'<Response><Play>https://{self.remote_host}/audio/{audio_key}</Play><Pause length="60"/></Response>'
        )
        time.sleep(duration)

    def hang_up(self, phone_operator):
        phone_operator.update(
            twiml=f'<Response><Hangup/></Response>'
        )

    def conversation(self, agent_a, talker_x):
        while agent_a.phone_operator is None:
            time.sleep(0.1)

        transcript_list = []
        data_to_write=""

        for i in range(3):
            text_a, audio_key, duration = agent_a.think_what_to_say(transcript_list)
            self.reply(agent_a.phone_operator, audio_key, duration)
            transcript_list.append(text_a)
            time.sleep(0.2)
            #data_to_write += f"[Cradle]\n {text_a} \n\n"

            text_b = agent_a.listen_and_transcribe(talker_x)
            transcript_list.append(text_b)
            #data_to_write += f"[Recipient]\n {text_b} \n\n"
            
            thinking_phrase = random.choice(agent_a.thinking_phrase_list)
            audio_key, duration = agent_a.text_to_audiofile(thinking_phrase)
            self.reply(agent_a.phone_operator, audio_key, duration)

        bye_txt = "I got it! Thank you! Good Bye!"
        audio_key, duration = agent_a.text_to_audiofile(bye_txt)
        #data_to_write += f"[Cradle]\n {bye_txt}"
        self.reply(agent_a.phone_operator, audio_key, duration)
        self.hang_up(agent_a.phone_operator)
       

        #call_context = agent_a.phone_operator
        #call_resource = call_context.fetch()
        #sid = call_resource.sid
        #latlng = self.sid2latlng[sid]
        #save_path = os.path.join(os.environ["LAST_CALL_DIR"], latlng+'.txt')
        #with open(save_path, 'w') as file:
        #    # Write the data to the file
        #    file.write(data_to_write)
        #print(f'Data has been written to {save_path}') 

    def start(self,):
        server = pywsgi.WSGIServer(
            ("", self.port), self.app, handler_class=WebSocketHandler
        )
        print("Server listening on: http://localhost:" + str(self.port))
        server.serve_forever()

    def run(self,):
        return self.app

def create_app():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    static_dir = "./static_dir"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    tws = FlaskCallCenter(remote_host=os.environ["REMOTE_HOST_URL"], port=5000, static_dir=static_dir)

    return tws.run()

if __name__ == '__main__':
    # force to use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    static_dir = "./static_dir"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    tws = FlaskCallCenter(remote_host=os.environ["REMOTE_HOST_URL"], port=2000, static_dir=static_dir)
    tws.start()
    
