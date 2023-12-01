from flask import Flask, render_template
from flask_socketio import SocketIO, send
import pandas as pd
import numpy as np
import openai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app, cors_allowed_origins="*")


df = pd.DataFrame(
    [
        {'building': 'Empire State', 'location': "NYC", 'height':1000},
        {'building': 'B', 'location': "LA", 'height':600},
        {'building': 'c', 'location': "Paris", 'height':900},
        {'building': 'C', 'location': "SF", 'height':800},
        {'building': 'P', 'location': "Dallas", 'height':500},
    ]
)
f = open("key.txt", "r")
key = f.read()
f.close()
f = open("prompt.txt", "r")
prompt = f.read()
f.close()
openai.api_key = key


@socketio.on('message')
def intakeMessage(msg):
    print("Message: " + msg)

    new_prompt = f"{prompt}\n\nQ:{msg} columns: {list(df.columns)}\n"
    print(f"New prompt: Q: {msg} columns: {list(df.columns)}\n")

    response = openai.Completion.create(engine='davinci',
                                        prompt=new_prompt,
                                        stop='\n',
                                        temperature=0,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        max_tokens=150
                                        )

    output = ''
    command = f"output = {response.choices[0].text.replace('A: ', '')}; dtype = type(output)"
    print("command:", command)
    ldict = {}
    exec(command, globals(), ldict)

    print('output:', ldict['output'])
    print('type:', ldict['dtype'])
    output = ldict['output']
    dtype = ldict['dtype']

    # convert to json serializable types
    if dtype == np.int64:
        output = int(output)



    print("output:", output)
    if dtype == pd.core.series.Series and len(output) == 1:
        command = f"output = {response.choices[0].text.replace('A: ', '')}.iloc[0]"
        exec(command)
        output = int(output)

    # can only send json serializable data types are allowed
    send(output, broadcast=True)  # If broadcast is false it will only send to one user (want to do that in the future)


@app.route('/')
def home():

    return render_template('test_frontend.html')


if __name__ == '__main__':
    socketio.run(app)