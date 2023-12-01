from flask import Flask , jsonify, request
import requests
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from waitress import serve

import json

GPT_MODEL = "gpt-3.5-turbo-0613"
openai.api_key = "your-token"


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )



app = Flask(__name__)
@app.route('/')
def home():
    return "running"

@app.route('/sentiment',methods = ["POST"])
def sentiment_sentence():

    data = request.get_json()

    text = data["text"]
    conversation = Conversation()
    conversation.add_message("user", f"Analyze the sentiment of following text and only output sentiment nothing else ex - positive , negative etc :{text}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)
    

@app.route('/named-entity',methods = ["POST"])
def named_entity():

    data = request.get_json()

    text = data["text"]
    conversation = Conversation()
    conversation.add_message("user", f"Do named entity recognition in the given sentence and also which type of entity is that please also provide and only output i want nothing extra explaination :{text}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/translation', methods = ["POST"])
def translation():

    data = request.get_json()

    text = data["text"]
    language = data["language"]

    conversation = Conversation()
    conversation.add_message("user",f"Translate this sentence to {language} only output the translation nothing extra : {text}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)


@app.route('/pos',methods = ["POST"])
def pos():

    data = request.get_json()

    text = data["text"]

    conversation = Conversation()
    conversation.add_message("user",f"Please provide the parts of speech tagging and only output for the {text}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/grammar_check',methods = ["POST"])
def grammar_check():

    data = request.get_json()

    text = data["text"]

    conversation = Conversation()
    conversation.add_message("user",f"Check the grammar and spelling and correct them return only originial sentence with corrected grammar and spelling  nothing extra : {text}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/summary',methods = ["POST"])
def summary():

    data = request.get_json()

    text = data["text"]

    conversation = Conversation()
    conversation.add_message("user",f"For the given text give best summary of the text as output nothing extra : {text}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/question-answer',methods = ["POST"])
def question_answer():

    data = request.get_json()
    question = data["question"]
    context = data["context"]
    conversation = Conversation()
    conversation.add_message("user",f"Answer the following question only output answer nothing extra: {question}\nContext: {context}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/offensive-detection',methods = ["POST"])
def offensive():

    data = request.get_json()

    text = data["text"]

    conversation = Conversation()
    conversation.add_message("user", f"Determine the category of offense in the given sentence. Choose from the following 5 categories and return only one category as output basically one word only:\n{text}")

    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/transliteration', methods = ["POST"])
def transliteration():

    data = request.get_json()

    text = data["text"]
    from_language = data["from_language"]
    to_language=data['to_language']

    conversation = Conversation()
    conversation.add_message("user", f"Transliterate the following sentence from {from_language} to {to_language}: \"{text}\"")

    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)

@app.route('/similarity-checker',methods = ["POST"])
def similarity_checker():

    data = request.get_json()
    text1 = data["text1"]
    text2 = data["text2"]
    conversation = Conversation()
    conversation.add_message("user", f"Calculate the similarity score should only return numerical value no explanation and used use embedding for the following two texts :\nText 1: {text1}\nText 2: {text2}")
    chat_response = chat_completion_request(
    conversation.conversation_history,
    # functions = None
)
    output = json.loads(chat_response.text)
    return jsonify(output)





if __name__ == "__main__":
    # app.run(debug = True,port=8081)

    serve(app, host='0.0.0.0', port=8080)