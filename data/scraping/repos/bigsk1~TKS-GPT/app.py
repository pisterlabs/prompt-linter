import os
import sys
import logging
from dotenv import load_dotenv
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import openai
from flask.logging import default_handler

# Load variables from the .env file
load_dotenv()

# Load your API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initializes the Flask app with the specified static folder for serving the React build
app = Flask(__name__, static_folder='chatbot-ui/build', static_url_path='')

# Enables Cross-Origin Resource Sharing for your Flask app, allowing your React app to make requests to the Flask server
CORS(app)

# Logging configuration
class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.url = request.url
        record.remote_addr = request.headers.get('X-Forwarded-For', request.remote_addr)
        record.method = request.method
        record.path = request.path
        record.user_agent = request.headers.get('User-Agent')
        return super().format(record)

file_handler = logging.FileHandler('flask_app.log')
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

formatter = RequestFormatter('%(asctime)s [%(levelname)s] %(remote_addr)s requested %(method)s %(path)s %(url)s\n%(message)s\nUser Agent: %(user_agent)s\n')

for handler in [file_handler, stdout_handler]:
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

app.logger.setLevel(logging.INFO)
app.logger.removeHandler(default_handler)

# Sets up the root route to serve the index.html file from the React build folder.
@app.route('/')
def index():
    app.logger.info('Serving index.html')
    return send_from_directory(app.static_folder, 'index.html')

# Sets up the /chat route to handle chat requests from the React app.
@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    chat_history = request.json["chat_history"]

    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for chat in chat_history:
            # Change the role from 'bot' to 'assistant' here
            role = chat["from"] if chat["from"] != "bot" else "assistant"
            messages.append({"role": role, "content": chat["message"]})
        messages.append({"role": "user", "content": message})
        
        print("Request messages:", messages)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=700,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        ai_message = response.choices[0].message["content"].strip()

    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        ai_message = "Error: API Error"

    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        ai_message = "Error: Connection Error"

    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        ai_message = "Error: Rate Limit Exceeded"

        # You can also add additional logic here to further process
        # the response before sending it to the front-end.
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        ai_message = f"Error: Unexpected Error - {e}"


    return jsonify({"message": ai_message})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
