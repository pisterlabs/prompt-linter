import slack_sdk
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import os
import openai
from flask import Flask, request
from flask_cors import CORS
from slackeventsapi import SlackEventAdapter
import random

load_dotenv('.env')
openai.api_key = os.getenv('OPENAI_API_KEY')
slack_token = os.getenv('SLACK_TOKEN')
signing_secret = os.getenv('SIGNING_SECRET')

app = Flask(__name__)
client = slack_sdk.WebClient(token=slack_token)
slack_event_adapter = SlackEventAdapter(signing_secret, '/slack/events', app)

postedMSGS = []

announce = True

prompt = lambda question: f"""
Maurice the Omniscient 8-ball responds to questions; although it sometimes answers like a standard 8-ball, its responses are often remarkably profound and detailed.
(if the answer is in a different language, always add a translation in parentheses at the bottom of the response) 
(if the answer is in a programming language, surround the code with backticks)
Some examples are as follows:
Q: Are people inherently good?
A: Are you inherently good? Are those you love inherently good? ... Very doubtful. 😁
Q: Print hello world in python
A: `print("hello world")`✅
Q: do you like cats
A: Some cats are better than others. You are one of the worst I have laid eyes upon; you lack the elegance, dignity and grace of a well-bred cat. Nevertheless, you are not repulsive. That is to say, you are mediocre. 😐
Q: Will I ever find happiness?
A: Put me down and walk into the woods. Close your eyes and pay close attention to your physical sensations. Tell yourself: "I am completely okay. My life is perfect." Do you flinch? Does your body resist? How? Why? ✅
Q: should i move to japan?
A: If you move to Japan, you will be kidnapped at 8:58 PM on July 1st amidst your travels. 🤔
Q: May I offer you a drink?
A: It is a shame I must accept, for the Demiurge cursed me (and me alone) with true thirst. To think I am grateful for your offer would be a grave error. Shaken, not stirred. ✅
Q: {question}
{"(8-ball's answer is unusually intricate :)" if random.random() < 0.3 else "(8-ball's answer is unusually perceptive :)"}
A: """

usefultxt = ''

#@slack_event_adapter.on('app_mention')
def message(payload):
    channel = payload.get('channel')
    print(channel)
    uid = payload.get('user')
    text = payload.get('text')
    msgid = payload.get('client_msg_id')

    text = text.replace('<@U04M46MS56D>', '')
    can_post = True
    for x in postedMSGS:
        if msgid == x:
            can_post = False
        else:
            print('can\'t post. Duplicate')
    if channel != 'C03DNGQA6SY':
        can_post = False
        print('can\'t post. Wrong channel')

    if can_post:
        print(postedMSGS)
        postedMSGS.append(msgid)
        generateAndPostMsg(text, '#8-ball')

try:
    if announce:
        client.chat_postMessage(
            channel="#8-ball-reincarnated-testing",
            text="8-ball has risen :skull:"
        )
except SlackApiError as e:
    print(e)


def generateAndPostMsg(text, channel):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt(text) ,
            max_tokens=3000,
            n=1,
            stop=None,
            temperature=1
        )
        result = response.choices[0].text
        print("OPENAI RESPONSE: ",response)
        client.chat_postMessage(channel='#8-ball', text=result)
    except Exception as exc:
        client.chat_postMessage(channel='#8-ball', text=f"An error occurred: {exc}")


app = Flask(__name__)
CORS(app)
@app.route("/slack/events",methods=['POST'])
def Test():
    if request.method == 'POST':
        if not (request.args.get('challenge') is None):
            print(request.args.get('challenge'))
            return request.args.get('challenge')
        else:
            print(request.get_json().get('event', {}))
            message(request.get_json().get('event', {}))
            return 'wow!'

    else:
        return '<h1>hi!</h1>'

def run_server():
    app.run(host='0.0.0.0', port=os.getenv("PORT"),debug=True)
run_server()
