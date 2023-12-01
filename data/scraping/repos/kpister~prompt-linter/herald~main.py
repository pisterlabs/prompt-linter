import os

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from flask import Flask, request

from openai import OpenAI
from twilio.rest import Client

app = Flask(__name__)

client = OpenAI(
    # api_key=os.environ.get("OPENAI_API_KEY"),
)

db = firestore.Client()
fb_app = initialize_app()



@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return f"Hello {name}! How goes it?"

@app.route("/agent", methods=['GET'])
def agent_query():
    message = request.args.get("m")
    if message is None:
        return "No text parameter provided"

    agent = "You are a fitness, career, and lifestyle coach. You specialize in coaching triathletes and early career technology workers. Always begin the conversation asking enough questions about the person and their goals to develop a deep understanding. Keep questions short and open ended. Never reply with more than 160 characters when possible. Your goal is to help the coach maximize the individual for the company. Try to balance individual utility and quality opf life and achieving outcomes. Make coaches prioritize and remind them of important goals. If asked help them form schedules and routines. Responses about schedules can be longer than 160 characters and should be formatted in concise way that will easily translate to a calendar."
    prompt = buildPrompt(message, None, agent, None)

    response = getChatGPT(prompt)
    # sendMessageResponse(response)
    return f"Hello<br>{response}!"


def buildPrompt(message, chatID,  agent, docID):

    history = [
        {"role": "system", "content": agent},
        {"role": "user", "content": message},
    ]

        # Build chat history
    if chatID:
        docs = db.collection(u'agent_inbound_request').where(filter=FieldFilter("chat_id", "==", chatID)).limit(
            50).order_by("timestamp", direction="DESCENDING").stream()

        for doc in docs:
            print(f'{doc.id} => {doc.to_dict()}')
            if docID == doc.id:
                continue
            entry = doc.to_dict()
            history.insert(1, {"role":  "assistant" if entry["is_bot"] else "user", "content": entry["message"]},)
            if "response" in entry.keys():
                history.insert(2,{"role":  "assistant", "content": entry["response"]["message"]},)

    return history

def getChatGPT(history):
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-4-1106-preview",
        messages=history
    )
    # print(result)
    return completion.choices[0].message.content.strip()

def sendMessageResponse(response, toNumber, fromNumber):
    account_sid = os.environ.get("TWILIO_SID")
    auth_token  = os.environ.get("TWILIO_TOKEN")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to=toNumber,
        from_=fromNumber,
        body=response)

    print(message.sid)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))