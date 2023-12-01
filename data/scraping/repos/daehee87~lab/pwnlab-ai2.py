import sys
from flask import Flask, request, jsonify
from datetime import date, datetime, timedelta
import time, json
import hashlib, os, base64, glob
from slackeventsapi import SlackEventAdapter
from slack_sdk.web import WebClient
import openai
import requests
import threading
import _thread

app = Flask(__name__)

# Our app's Slack Event Adapter for receiving actions via the Events API
#slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]
#slack_events_adapter = SlackEventAdapter(slack_signing_secret, "/slack/events", app)

# Create a SlackClient for your bot to use for Web API requests
slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
#slack_client = WebClient(slack_bot_token)

openai.api_key = os.environ["OPENAI_KEY"]

def ask_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # ChatGPT model
        prompt=prompt,
        max_tokens=500,  # Adjust the response length
        n=1,
        stop=None,
        temperature=0.7,  # Adjust creativity (lower value = more focused, higher value = more random)
        top_p=1,
    )

    return response.choices[0].text.strip()

def post_slack(channel_id, message, slack_token):
    data = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'token': slack_token,
            'channel': channel_id,
            'text': message
           }
    URL = "https://slack.com/api/chat.postMessage"
    res = requests.post(URL, data=data)
    return res

def next_day(day_number):
    today = datetime.today()
    today = today.replace(minute=0, second=0, microsecond=0)
    return today + timedelta(days=day_number)

def mk_key(t):
    key = str(t.year)
    key += str(t.month)
    key += str(t.day)
    key += str(t.hour)
    key += str(t.minute)
    return key

task = {}
def is_cmd(channel_id, cmd, token):
    global task
    opcode = 'save:'    # save:%d월%d일%d시%d분:이벤트명
    if cmd.startswith(opcode):
        try:
            when = cmd.split(":")[1].replace(" ", "")
            inner_dict = {}
            inner_dict['what'] = cmd.split(":")[2]
            inner_dict['channel'] = channel_id
        except:
            r = post_slack(channel_id, "명령 해석이 안됨.", token)
            print(r)
            return True
        
        t = date.today()
        current_year = t.year
        current_month = t.month
        t = None
        try:
            t = datetime.strptime(when, "%m월%d일%H시")
            t = t.replace(year = current_year)
            key = mk_key(t)
        except:
            t = None
        if t==None:
            try:
                t = datetime.strptime(when, "%m월%d일%H시%M분") 
                t = t.replace(year = current_year)
            except:
                t = None
        if t==None:
            try:                
                t = next_day(1)
                h = datetime.strptime(when, "내일%H시").hour
                t = t.replace(hour=h)
            except:
                t = None
        if t==None:
            try:                
                t = next_day(1)
                h = datetime.strptime(when, "내일%H시%M분").hour
                m = datetime.strptime(when, "내일%H시%M분").minute
                t = t.replace(hour=h)
                t = t + timedelta(minutes=m)
            except:
                t = None
        if t==None:
            try:
                t = datetime.strptime(when, "%d일뒤%H시")
                h = t.hour
                d = t.day
                t = next_day(d)
                t = t.replace(hour=h)
            except:
                t = None
        if t==None:
            try:
                t = datetime.strptime(when, "%d일뒤%H시%M분")
                h = t.hour
                d = t.day
                m = t.minute                
                t = next_day(d)
                t = t.replace(hour=h)
                t = t + timedelta(minutes=m)
            except:
                t = None

        if t==None:
            reply = "시간 해석 불가."
        else:
            key = mk_key(t)
            # time parsing OK
            if key in task:
                task[key]['what'] = task[key]['what'] + ", " + inner_dict['what']
                task[key]['channel'] = inner_dict['channel']
            else:
                task[key] = inner_dict
            reply = str(t) + " 에 [" + inner_dict['what'] + "] 기억함."

        r = post_slack(channel_id, reply, token)
        print(r)        
        return True

    # this is not a special command
    return False

def handle_msg(channel_id, msg, token):
    answer = ask_gpt(msg + '. 짧고 간결히 반말로 답해줘.')
    r = post_slack(channel_id, answer, token)
    print(r)

@app.route('/slack/events', methods=['POST'])
def handle_slack_events():
    global slack_bot_token
    # Load the request data as JSON
    request_data = json.loads(request.data)
    # Check if the event is a challenge event
    if 'challenge' in request_data:
        return jsonify({'challenge': request_data['challenge']})
    elif 'event' in request_data:
        event_data = request_data['event']
        # Check if the event is a message event
        if 'type' in event_data and event_data['type'] == 'app_mention':
            # Extract the message text
            message_text = event_data['text']
            if message_text.startswith('<'):
                idx = message_text.find('> ')
                if idx > 0:
                    message_text = message_text[idx+2:]
            print("Message received:", message_text)

            # Extract the channel ID
            channel_id = event_data['channel']
            print("Channel ID:", channel_id)
            
            if is_cmd(channel_id, message_text, slack_bot_token):
                return '', 200

            t = threading.Thread(target=handle_msg, args=(channel_id, message_text, slack_bot_token))
            t.start()
            return '', 200
        else:
            return '', 200
    else:
        return '', 200


def my_monitor(token):
    global task 
    print('monitor running!')
    while True:        
        # check if there is an event to notify
        time.sleep(5)
        try:
            t = datetime.today().replace(second=0, microsecond=0)
            key = mk_key(t)
            if key in task:
                event = task[key]['what']
                channel_id = task[key]['channel']
                msg = '리마인더: ' + event
                r = post_slack(channel_id, msg, token)
                print(r) 
                del task[key]
        except:
            r = post_slack(channel_id, "monitor has error!", token)
            print(r) 

t = threading.Thread(target=my_monitor, args=(slack_bot_token,))
t.start()

app.run(host='0.0.0.0', port=3000)

