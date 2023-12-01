import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, WebSocket

app = FastAPI()
from services.langchain.openai_messages import Messages
from services.langchain.openai_chat import OpenAIChatBot
from services.langchain.openai_image import OpenAIImage
from services.langchain.anthropic_chat import AnthropicChatBot
from server.create_ai_response import CreateAIResponse
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from authlib.integrations.flask_client import OAuth

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    while True:
        text_message = await websocket.receive_text()
        chat_handler(websocket, chat_message=text_message)
        await websocket.send_text(f"Message received: {text_message}")
        # Receive a request for an image from the client
        image_request = await websocket.receive_text()
        """DO a thing"""
        # Send the image data back to the client (placeholder for now)
        await websocket.send_text(f"Image request received: {image_request}")


async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    while True:
        # Receive a request for audio from the client
        audio_request = await websocket.receive_text()
        
        # Send the audio data back to the client (placeholder for now)
        await websocket.send_text(f"Audio request received: {audio_request}")

@app.websocket("/ws/image")
async def websocket_image_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    while True:

app = Flask(__name__)
create_ai_response = CreateAIResponse()

@app.route('/')
def route_index():
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api():
    """
    Returns a welcome message and instructions for using the HexLink API.
    Returns:
        str: The welcome message and instructions.
    """
    return """
    Welcome to the HexLink API
    Please make requests in the following format:
    POST /api/chat
    - Expects a JSON payload with the following keys:
    --    model: "The name of the model to use."
    --    content: "A message to send."
    --    role: "The role of the user sending the message."
    POST /api/image
    - Expects a JSON payload with the following keys:
    --    prompt: "The prompt to use."
    --    n: "The number of images to generate."
    --    size: "The size of the images to generate."
    """

@app.route('/api/chat', methods=['POST'])
def route_openai_chat():
    messages = Messages()
    data = request.get_json()
    print(data)
    message_id = data["payload"]["id"]
    mode = data["payload"]["model"]
    model = data["payload"]["model"]
    prompts = data["payload"]["prompts"]
    timestamp = data["payload"]["timestamp"]
    messages = messages.get_message(message=prompts)
    openai = OpenAIChatBot(model=model)
    ai_response = openai.get_chat_response(messages=messages)
    response = create_ai_response.create(ai_response=ai_response, mode="text")
    print(f"-- Response: \n{response}")
    return response

@app.route('/api/image', methods=['POST'])
def route_openai_image():
    data = request.get_json()
    print(data)
    prompt = data["payload"]["prompts"][0]
    n = data["payload"]["n"]
    size = data["payload"]["size"]
    openai = OpenAIImage()
    ai_response = openai.get_image_response(prompt=prompt, n=n, size=size)
    response = create_ai_response.create(ai_response=ai_response, mode="image")
    print(f"-- Response: \n{response}")
    return response

oauth = OAuth(app)

oauth.register(
    name='auth0',
    client_id=os.environ.get('CLIENT_ID'),
    client_secret=os.environ.get('CLIENT_SECRET'),
    authorize_url='https://dev-86o6o32adpcz7sj6.us.auth0.com/authorize',
    token_url='https://dev-86o6o32adpcz7sj6.us.auth0.com/oauth/token',
    api_base_url='https://dev-86o6o32adpcz7sj6.us.auth0.com/userinfo',
    client_kwargs={
        'scope': 'openid profile',
        'audience': 'hexlink_auth'
    }
)
@app.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    return oauth.auth0.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = oauth.auth0.authorize_access_token()
    session['token'] = token
    return redirect(url_for('profile'))

@app.route('/profile')
def profile():
    token = session.get('token')
    if token:
        resp = oauth.auth0.get('profile')
        profile_data = resp.json()
        return render_template('profile.html', profile=profile_data)
    else:
        return 'Access Denied'

if __name__ == '__main__':
    app.run(debug=True)