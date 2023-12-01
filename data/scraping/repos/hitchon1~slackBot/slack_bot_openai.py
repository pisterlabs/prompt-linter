from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai


app = App(token="xoxb")
openai.api_key = "sinset api openai"
chatbot = openai.Completion()


@app.event("app_mention")
def hello_command(ack, body, say):
    AI_prompt = str(body['event']['blocks'][0]['elements'][0]['elements'][1]['text'])
    response = chatbot.create(engine="text-davinci-003", prompt = AI_prompt, max_tokens=1000)
    say(str(response.choices[0].text))

if __name__ == "__main__":
    SocketModeHandler(app, "xapp-").start()
