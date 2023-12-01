import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from chatbot_engine import chat, create_index
from langchain.memory import ChatMessageHistory

load_dotenv()

index = create_index()

# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


def fetch_history(channel: str) -> ChatMessageHistory:
    bot_user_id = app.client.auth_test()["user_id"]
    conversation_history = app.client.conversations_history(
        channel=channel, limit=3)

    history = ChatMessageHistory()
    for message in reversed(conversation_history["messages"]):
        text = message["text"]
        if message["user"] == bot_user_id:
            history.add_ai_message(text)
        else:
            history.add_user_message(text)
    return history


@app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    history = fetch_history(channel)

    message = event["text"]
    bot_message = chat(message, history, index)
    say(bot_message)


# アプリを起動します
if __name__ == "__main__":
    app_env = os.environ.get("APP_ENV", "production")

    if app_env == "production":
        app.start(port=int(os.environ.get("PORT", 3000)))
    else:
        SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
