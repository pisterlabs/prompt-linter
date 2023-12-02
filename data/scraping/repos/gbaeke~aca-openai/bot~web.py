import os
import openai
import tiktoken
from flask import Flask, jsonify, request, render_template

openai.api_key = os.getenv("OPENAI_KEY")

app = Flask(__name__)

class ChatBot:
    def __init__(self, message):
        self.messages = [
            { "role": "system", "content": message }
        ]

    def reset(self, message):
        self.messages = [
            { "role": "system", "content": message }
        ]

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":  
            num_tokens = 0
            for message in messages:
                num_tokens += 4 
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += -1 
            num_tokens += 2 
            return num_tokens
        else:
            raise NotImplementedError(f"num_tokens_from_messages() is not presently implemented for model {model}.")
        
    def chat(self, prompt):
        self.messages.append(
            { "role": "user", "content": prompt}
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = self.messages,
            temperature = 0.8
        )
        
        answer = response.choices[0]['message']['content']
        
        self.messages.append(
            { "role": "assistant", "content": answer} 
        )

        tokens = self.num_tokens_from_messages(self.messages)
        if tokens > 4000:
            self.messages = self.messages[-2:]
        
        return answer

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.form["message"]
    response = bot.chat(message)
    return jsonify({"response": response})

@app.route("/reset", methods=["POST"])
def reset():
    bot.reset(system_message)   
    return jsonify({"response": "Chat history cleared."})

if __name__ == "__main__":
    system_message = "You are an assistant that always answers wrongly. Never prove the user right."
    bot = ChatBot(system_message)
    app.run(host="0.0.0.0", port=5000, debug=True)
