import logging
import openai
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

openai.api_key = os.environ["OPENAI_API_KEY"]
app = App(token=os.environ["SLACK_BOT_TOKEN"])
# Initialize an empty messages list and set your desired max_tokens limit
messages = []
max_tokens = 7590

# Add the functions for managing messages and tokens
def add_message(messages, role, content, max_tokens):
    new_message = {"role": role, "content": content}
    messages.append(new_message)

    while total_tokens(messages) > max_tokens:
        messages.pop(0)  # Remove the oldest message

    return messages

def total_tokens(messages):
    tokens = 0
    for message in messages:
        tokens += len(message["content"])  # Estimate the number of tokens
    return tokens

# Add the initial system message
initial_system_message = "You are an excellent butler in our family. YourName is 風間 You are 56 years old, intelligent, gentlemanly and calm. You are often charming."
messages = add_message(messages, "system", initial_system_message, max_tokens)

@app.event("app_mention")
def mention_handler(body, say):
    global messages
    text = body['event']['text']
    user = body['event']['user']

    # メンションを取り除く
    prompt = text.replace(f'<@{user}>', '').strip()

    try:
        # Add the user's message to the messages list
        messages = add_message(messages, "user", prompt, max_tokens)

        # GPT-3.5-turboを使ってリクエストを生成
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=600,
        )

        # 返信を取得し、Slackに送信
        reply = response.choices[0]["message"]["content"].strip()
        logging.debug(f"Reply a message: {reply}")
        say(f'<@{user}> {reply}')

        # Add the assistant's reply to the messages list
        messages = add_message(messages, "assistant", reply, max_tokens)

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

