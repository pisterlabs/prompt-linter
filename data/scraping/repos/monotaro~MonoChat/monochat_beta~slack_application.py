import re

import secret
from openai_client import OpenAiClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_client import SlackClient

secret = secret.get_secret()
SLACK_API_TOKEN = secret.get("SLACK_API_TOKEN")
SLACK_BOT_TOKEN = secret.get("SLACK_BOT_TOKEN")
SLACK_BOT_USER_ID = "XXXXXXXXXXX"  # BotのユーザID
SLACK_DELETE_REACTION = "del_monochat"  # 削除用のリアクションを作成しておく

app = App(token=SLACK_BOT_TOKEN)


def start_server():
    try:
        SocketModeHandler(app, SLACK_API_TOKEN).start()
    except KeyboardInterrupt:
        pass  # Ctrl+C が押された時はエラーではなく正常終了とする.


@app.event("app_mention")
def event_mention(event, say):
    """
    ボットがメンションされた場合に返信する.
    """
    open_ai_client = OpenAiClient()
    messages = _build_messages(event)
    try:
        response = open_ai_client.request(messages)
        say({"text": response, "thread_ts": event["ts"]})
    except Exception as e:
        say(
            {
                "text": "メッセージを処理できませんでした。もう一度実行してみてください。:monochat: :すまんな: \n※本メッセージはChat GPTからの返答ではありません。\n\nエラー詳細： ```"
                + str(e)
                + "```",
                "thread_ts": event["ts"],
            }
        )


@app.event("message")
def event_message(event, say):
    """
    Slack Apps に対する DM で発言された時に返信する.
    """
    # ボットとの DM 以外を除外する.
    if event.get("channel_type") != "im":
        return

    open_ai_client = OpenAiClient()
    messages = _build_messages(event)
    try:
        response = open_ai_client.request(messages)
        say({"text": response, "thread_ts": event["ts"]})
    except Exception as e:
        say(
            {
                "text": "メッセージを処理できませんでした。もう一度実行してみてください。:monochat: :すまんな: \n※本メッセージはChat GPTからの返答ではありません。\nエラー詳細： ```"
                + str(e)
                + "```",
                "thread_ts": event["ts"],
            }
        )


@app.event("reaction_added")
def message_delete(event):
    """
    削除用のリアクションがついた場合に発言を削除する.
    """
    if (
        event["reaction"] == SLACK_DELETE_REACTION
        and event["item_user"] == SLACK_BOT_USER_ID
    ):
        response = app.client.chat_delete(
            channel=event["item"]["channel"], ts=event["item"]["ts"]
        )


@app.command("/monochat")
def handle_slash_command(ack, client, command):
    """
    スラッシュコマンドを実行する.
    """
    ack()

    def post_ephemeral_message(text):
        client.chat_postEphemeral(
            channel=command["channel_id"], user=command["user_id"], text=text
        )

    # /monochat [text] の [text] 部分.
    split = command["text"].split(" ")
    subcommand, arguments = split[0], split[1:]

    # monochat の使い方を表示する.
    if subcommand == "help":
        post_ephemeral_message(
            _strip_heredoc(
                """
                    (1) MonoChat をメンションすると Azure OpenAI Service (ChatGPT) からのレスポンスが返されます。
                    (2) スレッド内でやり取りを繰り返すと、それまでの会話を考慮した回答がおこなわれます。
                    (3) スレッド内でもメンションは必要です。
                    (4) MonoChat との DM でも利用可能です。この場合はメンション不要です。
                    (5) MonoChat からの返信に :del_monochat: でリアクションすると返信を削除できます。
                """
            )
        )
        return

    post_ephemeral_message("Not Implemented")


def _build_messages(event) -> list[dict[str, str]]:
    """
    OpenAiClient に渡すメッセージを組み立てる.
    """
    if "thread_ts" in event:
        slack_client = SlackClient(SLACK_BOT_TOKEN)
        response = slack_client.get_replies(event["channel"], event.get("thread_ts"))
        messages = []
        for m in response["messages"]:
            # mにbot_idがあるかないか
            text = _remove_mention_string(m["text"])
            if "bot_id" in m:
                messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "user", "content": text})
            # スレッドの数が20を超えたら古いものから削除する
            if len(messages) > 20:
                messages.pop(0)
        return messages
    else:
        text = _remove_mention_string(event["text"])
        return [{"role": "user", "content": text}]


def _remove_mention_string(text: str) -> str:
    """
    テキストからメンション文字列を削除する.
    """
    return re.sub(r"<@.+?>", "", text, 1).strip()


def _strip_heredoc(text: str) -> str:
    return "\n".join(map(str.strip, text.splitlines())).strip()
