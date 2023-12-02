import os

import gradio as gr
from chatbot_engine import chat, create_index
from dotenv import load_dotenv

from langchain.memory import ChatMessageHistory


def respond(message, chat_history):
    history = ChatMessageHistory()
    for [user_message, ai_message] in chat_history:
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)

    bot_message = chat(message, history, index)
    chat_history.append((message, bot_message))
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    load_dotenv()

    app_env = os.environ.get("APP_ENV", "production")

    if app_env == "production":
        username = os.environ["GRADIO_USERNAME"]
        password = os.environ["GRADIO_PASSWORD"]
        auth = (username, password)
    else:
        auth = None

    index = create_index()

    demo.launch(auth=auth)
