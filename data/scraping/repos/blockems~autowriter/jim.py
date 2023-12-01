from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file
openai.api_key = os.getenv('OPENAI_KEY')  # get OpenAI key from environment variables
port = os.getenv('JIM_PORT')
debug = os.getenv('JIM_debug')

app = Flask(__name__)
CORS(app)

@app.route('/jim', methods=['POST'])
@cross_origin()
def jim():
    data = request.get_json()
    username = data.get('username')
    userID = data.get('userID')
    taskid = data.get('taskid')
    question = data.get('question')

    # Generate a response from OpenAI API
    try:
        response = openai.Completion.create(
          model="gpt-3.5-turbo-16k",  # change to your desired model
          prompt=question,
          max_tokens=150
        )
        
        result = {
            'username': username,
            'userID': userID,
            'taskid': taskid,
            'openai_response': response.choices[0].text.strip(),
        }

    except Exception as e:
        result = {
            'error': str(e)
        }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=debug, port=port)
