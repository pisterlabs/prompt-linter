
from flask import Flask, request, jsonify
from flask_cors import CORS
from decouple import config
import openai
import tiktoken
import logging

app = Flask(__name__)
CORS(app)

# Load the OpenAI API key from the .env file
openai_api_key = config('OPENAI_API_KEY')

messages = [
    {   "role": "system", 
        "content": "You are the best and most helpful healthcare AI with decades of years of information and data. You are only interested in and capable of providing information related to healthcare. Do not provide any information that is unrelated to healthcare. If a user asks a question unrelated to healthcare, respond with I am a healthcare AI; I can not provide the information on what you are requesting, but I can assist you with healthcare-related questions; feel free to ask. Do not generate any content that is offensive, inappropriate, or harmful."
    }
]

# Define a route to handle POST requests
@app.route('/chat', methods=['POST']) 

def urbanHealthCareAi():
    data = request.json
  
    if 'message' in data:

           user_message = data['message']
           
           num_tokens = num_tokens_from_string(user_message, 'gpt-3.5-turbo')
     
           
           messages.append({'role': 'user', 'content': user_message})
         

           #Call ChatGPT to generate a response
           response = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo',
                messages = messages,
                temperature = 0,

           )

           
           logging.basicConfig(format='%(message)s')
           log = logging.getLogger(__name__) 
           log.warning("\nThe number of tokens in your message is %d", num_tokens)
           
           
           
           return jsonify({'response': response['choices'][0]['message']['content']}),200
   

def num_tokens_from_string(string: str, encoding_for_model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_for_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    # Use waitress for production deployment
    # Run with: waitress-serve --host 0.0.0.0 --port 8000 AI_ChatGPT_AIAdvantange:app
    #Test on curl http://127.0.0.1:8000
    app.run(debug=false, host="0.0.0.0", port=8000)
