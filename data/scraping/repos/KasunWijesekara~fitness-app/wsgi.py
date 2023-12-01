import os
from dotenv import load_dotenv

from flask import Flask, request, jsonify
import requests

# Your provided imports
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

app = Flask(__name__)

# Your provided initializations
global_chat_history = ""

template = """You are a helpful assistant who helps the user and also remembers the chat history.
CHAT-HISTORY:
{history}    
USER: {human_input}
AI: 
"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=50,
        openai_api_key=openai_api_key,
    ),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(memory_key="history", k=2),
)


def generate_answer(question, history):
    response = chatgpt_chain({"human_input": question, "history": history})

    if "text" in response:
        answer = response["text"]
    else:
        print("Response does not have a 'text' key. Full response:", response)
        answer = "Sorry, I can't help with that right now."

    return answer


@app.route("/webhook", methods=["GET"])
def verify():
    """This endpoint is used by Facebook to verify your webhook."""
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Invalid verification token"


@app.route("/webhook", methods=["POST"])
def messenger_webhook():
    global global_chat_history
    data = request.json

    if data["object"] == "page":
        for entry in data["entry"]:
            for messaging in entry["messaging"]:
                sender_id = messaging["sender"]["id"]

                if "message" in messaging:
                    incoming_que = messaging["message"]["text"].lower()
                    print("Question: ", incoming_que)

                    # Generate the answer using GPT-3
                    answer = generate_answer(incoming_que, global_chat_history)
                    global_chat_history += (
                        f"\nUser: {incoming_que}\nAi: {answer}"  # Update the history
                    )
                    print("BOT Answer: ", answer)

                    send_message(sender_id, answer)
    return "OK", 200


def send_message(recipient_id, message_text):
    """Send message to user using Facebook Graph API."""
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {"recipient": {"id": recipient_id}, "message": {"text": message_text}}
    r = requests.post(
        "https://graph.facebook.com/v13.0/me/messages",
        params=params,
        headers=headers,
        json=data,
    )
    if r.status_code != 200:
        print("Failed to send message: " + r.text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
