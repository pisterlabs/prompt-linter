import json
import os
from pathlib import Path
from re import template
from typing import Text
from summarizer_service import get_summary
import slack
from dotenv import load_dotenv
from flask import Flask, Response, request
from flask_cors import CORS
from slackeventsapi import SlackEventAdapter
import logging
import openai
from summarizer_service import SummaryReport
import threading
from summarizer_service import suggest_reply_to_conversation
from summarizer_service import train_engine

logging.basicConfig(filename="logs/squasher_service.log",
                            format='%(asctime)s %(message)s',
                                                filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
app = Flask(__name__)
CORS(app)
signing_secret = os.environ['SIGNING_SECRET']
slack_token = os.environ['SLACK_TOKEN']

slack_event_adapter = SlackEventAdapter(
            signing_secret, "/slack/events", app
            )
client = slack.WebClient(slack_token)
users = { 'UL9MC3NJC' : 'Francis' , 'U016DGCKK0C' : 'Apoorva', 'U1D0YS3HC' : 'Varun' }

# Request
#{
#    content: string,
#    summary_lines: int
#}
# Responds with a summary of the content
@app.route("/squashit", methods=["POST"])
def summarize():
    try:
        data = request.form
        logger.info(data)
        summary = get_summary(data["content"], data["summary_lines"])
        logger.info(summary)
        return Response(summary), 200
    except Exception as e:
        logger.error(e)
        return Response(), 500

#Slack bot request that reads the messages of a thread and replies back with a short summary
@app.route("/slack-summarize", methods=["POST"])
def slack_summarize():
    slack_request = request.form
    # starting a new thread for doing the actual processing
    x = threading.Thread(
            target=process_slack_summarize,
            args=(slack_request,)
    )
    x.start()
    return Response(), 200


def process_slack_summarize(slack_request):
    try:
        payload = json.loads(slack_request.get('payload'))
        #logger.info(payload)
        channel_id = payload['channel']['id']
        message = payload['message']
        user_id = payload['user']['id']
        logger.info("Request by %s in channel %s" %(user_id, channel_id))
        if "thread_ts" not in message:
                client.chat_postEphemeral(channel=channel_id, text=f"Sorry: We can only summarize a thread", user=user_id)
        else:
                thread_ts = payload['message']['thread_ts']
                replies = client.conversations_replies(channel=channel_id, ts=thread_ts).get('messages')
                conversation = ''
                for reply in replies:
                    user_id = reply.get('user')
                    user_details_for_text = client.users_profile_get(user=user_id)
                    text = ""
                    text += user_details_for_text["profile"]["display_name_normalized"] + " says: "
                    text += reply.get('text')
                    conversation +=  text + ". "
                summary_report = get_summary(conversation, 2, "Below is a conversation")
                summary = summary_report.get_summary()
                response = client.chat_postEphemeral(channel=channel_id, text=summary, thread_ts=thread_ts, user=user_id)
                logger.info(response)
    except Exception as e:
        logger.error(e)


#Receives feedback that can be used to train data
@app.route("/feedback", methods=["POST"])
def feedback():
        data = request.form
        logger.info(data)
        content = data["content"]
        original_summary = data["original_summary"]
        suggested_summary = data["suggested_summary"]
        engine = data["engine"]
        train_engine(content, original_summary, suggested_summary, engine)
        return Response(), 200


@app.route("/health-check", methods=["GET"])
def health_check():
        return Response(), 200

# Gives back suggestions on the content based on the request in the suggestion
#{
#  "content" : "Full mail thread"
#  "suggestion" : "Suggest possible responses for 'Person who has to respond' "
#}
@app.route("/suggest-reply", methods=["POST"])
def suggest_reply():
        data = request.form
        logger.info(data)
        response = suggest_reply_to_conversation(data["content"], data["suggestion"])
        logger.info(response)
        return Response(response), 200


if __name__ == "__main__":
        app.run(debug=True, host="0.0.0.0", port=7000)

