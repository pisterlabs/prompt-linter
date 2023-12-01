import audioop
import base64
import json
import os
from flask import Flask, request, jsonify
from flask_sock import Sock, ConnectionClosed
from twilio.twiml.voice_response import VoiceResponse, Start, Dial, Client
from twilio.rest import Client as TwilioClient
import vosk
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
from flask_cors import CORS
import cohere
import requests
import random
import json
import time


predictions = {}
app = Flask(__name__)
CORS(app)
sock = Sock(app)
twilio_client = TwilioClient()
model = vosk.Model('model')
public_url = "https://listenlink.herokuapp.com"
# else:
#     from pyngrok import ngrok
#     port = 5002
#     public_url = ngrok.connect(port, bind_tls=True).public_url
print(public_url)
number = twilio_client.incoming_phone_numbers.list()[0]
number.update(voice_url=public_url + '/call')
print(f'Waiting for calls on {number.phone_number}')
CL = ""



TEST_IDENTITY = 'user'

@app.route("/")
def index():
    return "hello, World", 200

def get_predictions(values):
    global predictions
    co = cohere.Client(os.environ["COHERE_KEY"])
    predictions = co.classify(model="87434abe-cca3-4825-8f05-53657e3e9bae-ft",
                       inputs=[values])
    main_prediction = predictions[0].prediction # this will eventually be returned in the response, have yet to deal with it
    predictions = {k: v[0] for k, v in predictions[0].labels.items()}
    predictions["prediction"] = main_prediction


@app.route("/predictions", methods=["GET"])
def test():
    global predictions
    return predictions, 200

       #  'The confidence levels of the labels are: {}'.format(
       # response.classifications) , 200

@app.route('/call', methods=['POST'])
def call():
    """Accept a phone call."""
    response = VoiceResponse()
    start = Start()
    start.stream(url=f'wss://{request.host}/stream')
    response.append(start)
    # response.say('Please leave a message')
    # response.pause(length=60)
    dial = Dial()
    client = Client()
    client.identity(TEST_IDENTITY)
    dial.append(client)
    response.append(dial)
    print(f'Incoming call from {request.form["From"]}')
    return str(response), 200, {'Content-Type': 'text/xml'}

@app.route('/twilio-token', methods=["GET"])
def get_token():
# required for all twilio access tokens
# To set up environmental variables, see http://twil.io/secure
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    api_key = os.environ['TWILIO_API_KEY']
    api_secret = os.environ['TWILIO_API_KEY_SECRET']

    # required for Chat grants
    # Create access token with credentials
    token = AccessToken(account_sid, api_key, api_secret, identity=TEST_IDENTITY)

    # Create a Voice grant and add to token
    voice_grant = VoiceGrant(
        incoming_allow=True,  # Optional: add to allow incoming calls
    )
    token.add_grant(voice_grant)

    # Return token info as JSON
    return str(token.to_jwt()), 200


@sock.route('/stream')
def stream(ws):
    """Receive and transcribe audio stream."""
    initial_time = -1
    rec = vosk.KaldiRecognizer(model, 16000)
    while True:
        message = ws.receive()
        packet = json.loads(message)
        if packet['event'] == 'start':
            print('Streaming is starting')
        elif packet['event'] == 'stop':
            print('\nStreaming has stopped')
        elif packet['event'] == 'media':
            audio = base64.b64decode(packet['media']['payload'])
            audio = audioop.ulaw2lin(audio, 2)
            audio = audioop.ratecv(audio, 2, 1, 8000, 16000, None)[0]
            if rec.AcceptWaveform(audio):
                r = json.loads(rec.Result())
                # if r is words, then toss into classify and return main classification with statistical value
                # print(r['text'])
                val = r['text']
                val = val.strip()
                if (val!= "" and (time.time() - initial_time) > 5):
                    initial_time = time.time()
                    print(val)
                    get_predictions(val)
                elif val=="":
                    print("empty string")
                else:
                    print("timeout")
            # else:
                # r = json.loads(rec.PartialResult())
                # if r is words, then toss into classify and return main classification with statistical value
                # print(CL + r['partial'] + BS * len(r['partial']), end='', flush=True)


if __name__ == '__main__':

    app.run()
