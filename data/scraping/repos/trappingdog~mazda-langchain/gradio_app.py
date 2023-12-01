import os

import gradio as gr
from chatbot_engine import chat, create_index
from dotenv import load_dotenv

from langchain.memory import ChatMessageHistory


# GradioのHow to Create a Chatbotを参照 https://gradio.app/creating-a-chatbot/
# LangChainのChatMessageHistoryを参照 https://python.langchain.com/docs/modules/memory/
def respond(message, chat_history): # Chat履歴を渡す / API利用時履歴実装に注意
    history = ChatMessageHistory() # ChatMessageHistoryインスタンス作成
    for [user_message, ai_message] in chat_history:
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)

    bot_message = chat(message, history, index)
    chat_history.append((message, bot_message))
    return "", chat_history


# Gradioレイアウト、画像カラムとチャットカラムを実装
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image("src/firefly_woman.jpg", label='', height=400)
        with gr.Column(scale=1):
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__": # このファイルが入り口として実行された場合
    load_dotenv()
    
    app_env = os.environ.get("APP_ENV", "production")

    if app_env == "production":
        username = os.environ["GRADIO_USERNAME"]
        password = os.environ["GRADIO_PASSWORD"]
        auth = (username, password)
    else:
        auth = None

    index = create_index() # chatbot_engine.pyより

    demo.launch(auth=auth)
