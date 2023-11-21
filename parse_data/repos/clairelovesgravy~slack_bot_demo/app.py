import os
from flask import Flask, request
from dotenv import load_dotenv, find_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

# Load the environment variables
load_dotenv(find_dotenv())

slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]
signing_secret = os.environ["SIGNING_SECRET"]

app = App(token=slack_bot_token, signing_secret=signing_secret)

flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

CHATAI = ChatOpenAI(temperature=0.9,model_name='gpt-3.5-turbo',max_tokens=1500, openai_api_key=openai_api_key)

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

memory = ConversationBufferWindowMemory(k=3, memory_key="history")

chat_chain = ConversationChain(memory=memory, llm=CHATAI, prompt=prompt,verbose=True)


@app.event("app_mention")
def handle_app_mentions(body, say, logger):
    user_id = body["event"]["user"]
    say(f"Hi there, <@{user_id}>!")

    text = body["event"]["text"]
    response = chat_chain.predict(input=text)
    say(response)

@app.event("message")
def handle_message_events(body, say, logger):
    user_id = body["event"]["user"]
    say(f"Hi there, <@{user_id}>!")

    text = body["event"]["text"]
    response = chat_chain.predict(input=text)
    say(response)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run(port=5001)