from flask import Flask, render_template, session, copy_current_request_context
from flask_socketio import SocketIO, emit, disconnect
from threading import Lock
from GPTresponse import messages, OpenAIKey
import time
import openai
import re

openai.api_key = OpenAIKey
model_engine = "gpt-3.5-turbo"

async_mode = None
app = Flask(__name__,template_folder='./templates',static_folder='./templates/static')
app.config['SECRET_KEY'] = 'your_secret_key' # This is supposed to be newly generated for each session but brand tech
socket_ = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

@app.route('/')
def index():
    return render_template('index.html', async_mode=socket_.async_mode)


@socket_.on('my_event', namespace='/test')
def test_message(message):
    
    session['receive_count'] = session.get('receive_count', 0) + 1

    messages.append({"role": "user", "content": message['data']})
    
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
        temperature=0.7
    )

    # Parse the response and output the result
    output_text = response['choices'][0]['message']['content']

        # Append assistant's response to the messages
    messages.append({"role": "assistant", "content": output_text})
    print("EMITTING RESPONSE | ",output_text)
    emit('my_response',
        {'data':output_text, 'count': session['receive_count']})
    
@socket_.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)


if __name__ == '__main__':
    socket_.run(app, debug=True)
