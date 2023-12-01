from flask import Flask, render_template, request
from openai import OpenAI, ChatCompletion
import uuid
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    model_name = "gpt-3.5-turbo"  # OpenAIのGPT-3.5-turboモデルを使用
    if request.method == 'POST':
        session_id = request.form.get('sessionId')
        user_message = request.form.get('user')

        # Create the chat model and get the response
        chat_response = ChatCompletion.create(
          model=model_name,
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        
        send_messages_response = chat_response['choices'][0]['message']['content']
        
        # pass session_id and chat_response back to the template here
        return render_template('index.html', session_id=session_id, send_messages_response=send_messages_response)

    new_uuid = str(uuid.uuid4())
    send_messages_response = "How can I assist you today?"
    return render_template('index.html', session_id=new_uuid, send_messages_response=send_messages_response)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
