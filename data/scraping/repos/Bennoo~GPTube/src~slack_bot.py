import os

from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.error import BoltUnhandledRequestError

from slack_functions import slack_helper
from langchain_functions import langchain_helper
from langchain_functions.custom_chain import waiting_time

from langchain.embeddings.openai import OpenAIEmbeddings

app = slack_helper.get_slack_bolt_app_azure('gpt4-32k', 'gpt35t', 0.6)
# app = slack_helper.get_slack_bolt_app_azure('gpt35t', 'gpt35t', 0.1)
embeddings = OpenAIEmbeddings(deployment="embedding", chunk_size=16)
# embeddings = OpenAIEmbeddings()

@app.message()
def on_message(body, message, say):
    waiting_time_generator = waiting_time.get_azureo_waiting_time_generator()
    user = message['user']
    channel_id = message["channel"]
    if app.document_db is None:
        warning_text = waiting_time_generator.run('no video set, you should give me a valid youtube video link.')
        slack_helper.say_standard_block_answer_message(say, answer=warning_text, channel_id=channel_id)
    else:
        waiting_text = waiting_time_generator.run(
            "Give me some time, I will check the video content"
        )
        response = say(
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": f"<@{user}> _{waiting_text}_"}}
            ],
            text=f"{waiting_text}",
            channel_id=channel_id
        )
        text_question = body["event"]["text"]
        answer, generated_question = langchain_helper.get_response_qa_from_query_bolt(text_question, app, "stuff")

        text = generated_question
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_ <@{user}> asked for:_ *{generated_question}*",
                },
            }
        ]

        app.client.chat_update(
            channel=response["channel"], ts=response["ts"], text=text, blocks=blocks
        )

        slack_helper.say_standard_block_answer_message(
            say, answer=answer, exchanges=len(app.chat_history), channel_id=channel_id
        )

@app.command("/set_video")
def repeat_text(ack, say, command):
    # Acknowledge command request
    ack()
    url = command['text']
    user = command['user_id']
    waiting_time_generator = waiting_time.get_azureo_waiting_time_generator()
    waiting_text = waiting_time_generator.run(
        "Watching the whole video... It takes a moment.."
    )
    say(text=waiting_text)
    db, meta_data = langchain_helper.set_video_as_vector(url, embeddings)
    app.document_db = db
    app.meta_data = meta_data
    say(text=f"Video is set by <@{user}>! {url}")

@app.action('button-clear')
def on_clear(ack, say):
    app.chat_history = []
    app.document_db = None
    app.meta_data = None
    ack()
    clear_text = "All is cleared"
    say(
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"_..{clear_text}.._"}
            }
        ],
        text=f"{clear_text}")

@app.error
def handle_errors(error, say):
    if isinstance(error, BoltUnhandledRequestError):
        return
    else:
        say(text='Something goes wrong in the slack app...')

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
