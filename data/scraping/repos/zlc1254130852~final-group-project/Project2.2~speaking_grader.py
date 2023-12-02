from flask import Blueprint
from flask import render_template, request
from setting import socketio
from transform import pcm2wav
from AI_chat import openai_client
from api_key import DIFY_API_KEY_SPEAKING
from stream_generation import generate_dify_speaking
import base64, json
from assess2 import send_to_assess

import requests

speaking_grader_bp = Blueprint('speaking_grader', __name__)

file3 = []

@speaking_grader_bp.route('/upload',methods=['GET'])
def upload():
    return render_template("upload.html")

@speaking_grader_bp.route('/file',methods=['POST'])
def save_file():
    data = request.files

    file = data['file']

    buffer_data = file.read()
    with open("static/en/"+file.filename, 'wb+') as f:
        f.write(buffer_data)

    form=1
    if file.filename[-4:]==".pcm":
        form=1
    elif file.filename[-4:]==".wav":
        form=2
    elif file.filename[-4:]==".mp3":
        form=3

    result=send_to_assess(base64.b64encode(buffer_data).decode(),form)
    print(result)

    f.close()
    audio_file = open("static/en/"+file.filename, "rb")

    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    audio_file.close()
    print(transcript)

    socketio.emit('reply5', transcript+"\n")

    socketio.emit('reply5', 'PronAccuracy: ' + str(json.loads(result)['PronAccuracy']) + "\nPronFluency: " + str(
        json.loads(result)['PronFluency'])+"\n\nWaiting for the response...\n\n")

    headers = {'Authorization': DIFY_API_KEY_SPEAKING, 'Content-Type': 'application/json'}
    payload = {'inputs': {}, 'query': transcript, 'response_mode': 'streaming',
               'conversation_id': '', 'user': 'abc-123'}

    response = requests.post("https://api.dify.ai/v1/chat-messages", data=json.dumps(payload),
                             headers=headers)
    generate_dify_speaking(response)

    return ""

@socketio.on('play3')
def play3(req):
    file3.append(open("static/en/example3.pcm", "wb"))

@socketio.on('update3')
def update3(data):
    if not file3:
        return

    if type(data)==dict:

        file_popped=file3.pop()
        file_popped.close()

        file_path = r"static/en/example3.pcm"
        to_path = r"static/en/example3.wav"
        pcm2wav(file_path, to_path)

        audio_file = open("static/en/example3.wav", "rb")

        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        socketio.emit('reply5', transcript+"\n")

        if 'type' in data:
            with open("static/en/example3.pcm", "rb") as f:
                base64_data = base64.b64encode(f.read()).decode()
                print(type(base64_data))
                result = send_to_assess(base64_data,1)
                socketio.emit('reply5', 'PronAccuracy: ' + str(json.loads(result)['PronAccuracy']) + "\nPronFluency: " + str(
        json.loads(result)['PronFluency'])+"\n\nWaiting for the response...\n\n")

            headers = {'Authorization': DIFY_API_KEY_SPEAKING, 'Content-Type': 'application/json'}
            payload = {'inputs': {}, 'query': transcript, 'response_mode': 'streaming',
                       'conversation_id': '', 'user': 'abc-123'}

            response = requests.post("https://api.dify.ai/v1/chat-messages", data=json.dumps(payload),
                                     headers=headers)

            generate_dify_speaking(response)

        else:

            socketio.emit('reply3', {"result": " "+transcript, "end": 1})
    else:
        file3[-1].write(data)