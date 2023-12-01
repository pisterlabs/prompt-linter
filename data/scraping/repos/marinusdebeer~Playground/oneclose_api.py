from flask import Flask, jsonify, request
import openai
import os
from flask_cors import CORS
from flask_socketio import SocketIO, emit, send
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False
# socketio = SocketIO(app, transports=['websocket', 'polling'])
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)
# message_history = [{"role": "system", "content": f"You are a chat bot to be used for financial mortgages. The company you work at is called Oneclose Inc. At all times be respectful."}]
prompt_history = []
user_sids = {}
@socketio.on('connect')
def handle_connect():
    profile_id = request.args.get('profileId')
    if profile_id:
        user_sids[profile_id] = request.sid
        print(f"User connected: {request.sid}, profileId: {profile_id}")

@socketio.on('disconnect')
def on_disconnect():
    profile_id = request.args.get('profileId')
    if profile_id and profile_id in user_sids:
        del user_sids[profile_id]
        print(f"User disconnected: {request.sid}, profileId: {profile_id}")
def text(input):
    # message_history.append({"role": "user", "content": f"{input}"})
    # model="gpt-3.5-turbo",#10x cheaper than davinci, and better. $0.002 per 1k tokens
    messages = [{"role": "system", "content": f"You are a chat bot to be used for financial mortgages. The company you work at is called Oneclose Inc. At all times be respectful."}]
    messages.extend(input.messages)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        max_tokens=1000
    )
    reply_content = completion.choices[0].message.content
    return reply_content

def text_stream(input):
    # message_history.append({"role": "user", "content": f"{input}"})
    # model="gpt-3.5-turbo",#10x cheaper than davinci, and better. $0.002 per 1k tokens
    messages = [{"role": "system", "content": f"You are a broker to be used for financial mortgages. The company you work at is called Oneclose Inc, so speak from the perspective of this company at all times."}]
    # print(input['messages'])
    # input['messages'][-1].content = "in less than 400 words: " + input['messages'].content
    messages.extend(input['messages'])
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        # max_tokens=500,
        stream=True
    )

    try:
      for chunk in completion:
        if 'content' in chunk.choices[0].delta:
          print(chunk.choices[0].delta.content)
          socketio.emit('message', chunk.choices[0].delta.content, room=user_sids.get(str(input['user'])))
    except Exception as e:
      print('error', e)
    return ''
def code_stream(input):
  message_history.append({"role": "user", "content": f"{input}"})
  # model="gpt-3.5-turbo",#10x cheaper than davinci, and better. $0.002 per 1k tokens
  completion = openai.Completion.create(
      model="text-davinci-003", 
      prompt=input,
      max_tokens=300,
      stream=True
  )
  try:
    for chunk in completion:
      print(chunk.choices[0].text)
      socketio.send(chunk.choices[0].text)
  except Exception as e:
    print('error', e)
  return ''
def code(input):
  message_history.append({"role": "user", "content": f"{input}"})
  # model="gpt-3.5-turbo",#10x cheaper than davinci, and better. $0.002 per 1k tokens
  completion = openai.Completion.create(
      model="text-davinci-003", 
      prompt=input,
      max_tokens=300
  )
  print(completion.choices[0].text)
  return completion.choices[0].text
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400
    prompt = data['prompt']
    try:
        # message = predict_text(prompt)
        message = text_stream(prompt)
        return jsonify({'message': message}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error processing the request'}), 500


if __name__ == '__main__':
    # app.run(debug=True, port=os.getenv('PORT', 3000))
    socketio.run(app, port=3000, debug=True)


