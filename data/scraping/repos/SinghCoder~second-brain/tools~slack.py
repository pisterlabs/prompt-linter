import os

from dotenv import load_dotenv
from langchain.tools import tool
from slack_sdk import WebClient

load_dotenv()

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
slack_user_id = os.environ.get("SLACK_USER_ID")

def send_message_in_thread(channel_id, thread_ts, message):
    print(f"Sending slack message: {message} at channel: {channel_id} in thread: {thread_ts}")
    try:
        resp = slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=message)
    except Exception as e:
        print(f"Error sending slack message: {e}")
        return {"error": {e}}
    return None

def get_slack_message_link_from_ts(ts, channel_id = "C05NCPSEY0N"):
    try:
        resp = slack_client.chat_getPermalink(channel=channel_id, message_ts=ts)
        return resp['permalink']
    except Exception as e:
        print(f"Error getting slack message link: {e}")
        return {"error": {e}}

def send_slack_dm(message="", blocks=[]):
    try:
        resp = slack_client.chat_postMessage(channel=slack_user_id, text=message, blocks=blocks)
    except Exception as e:
        print(f"Error sending slack message: {e}")
        return {"error": {e}}
    return None

def send_slack_message(channel_id, message):
    print(f"Sending slack message: {message}")
    meeting_titles = []
    meetings = message.split("today: ")[1].split(" conflicts with ")
    print(meetings)
    meeting_titles = [m.split(" at ")[0] for m in meetings]
    print(meeting_titles)
    try:
        resp = slack_client.chat_postMessage(channel=channel_id, text=message, blocks=[
            {
			"type": "section",
			"text": {
				"type": "mrkdwn",
                "text": f"Found conflicts in the meetings: \n\n*{meeting_titles[0]}* conflicts with *{meeting_titles[1]}*"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": f"Cancel {meeting_titles[0]}"
                        },
                        "style": "primary",
                        "value": "click_me_123",
                        "url": "https://calendar.google.com"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": f"Cancel {meeting_titles[1]}"
                        },
                        "value": "click_me_123",
                        "url": "https://calendar.google.com"
                    }
                ]
            }
        ])
    except Exception as e:
        print(f"Error sending slack message: {e}")
        return {"error": {e}}
    return None

@tool
def notify_user(message):
    """
    Notify the user on slack with given message.
    """
    return send_slack_message(channel_id=slack_user_id, message=message)
