import os
import shutil

from app_modules.overwrites import postprocess
from app_modules.presets import *
from check_then_answer import ask_gpt, query_vector_store
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


embedding_model_name = "models/embedding_models/text2vec"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)


def predict(query, history):
    vector_store_path = "data/vector_stores/laws_vector_store"
    retrieved_laws = query_vector_store(query=query, vector_store_path=vector_store_path, embeddings=embeddings)
    input = (
        f"问题：{query} \n "
        f"为了回答这个问题，我们检索到相关法条如下：\n"
        f"{''.join(retrieved_laws)}\n"
        f"利用以上检索到的法条，请回答问题：{query}\n"
        f"要求逻辑完善，有理有据，不允许伪造事实。"
    )
    completion = ask_gpt(input=input)
    if history == None:
        history = []
    history.append((query, completion))
    chatbot = history
    print("答案：", completion)
    message = ""

    return [message, chatbot, state, "\n-----------------------------\n\n".join(retrieved_laws)]


with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    gr.Markdown("""<h1><center>AssistantLawer-大模型法律助手</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown([
                "text2vec-large"
            ],
                label="Embedding model",
                value="text2vec-large")


        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='AssistantLawyer').style(height=400)
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
            with gr.Row():
                send = gr.Button("🚀 发送")
            with gr.Row():
                gr.Markdown("""提醒：<br>
                                        AssistantLawyer 是基于深度学习技术构建的，它可以提供有价值的法律建议和解释，但不应视为法律专家的替代品。在重要的法律事务中，建议您咨询专业的法律顾问或律师。 <br>
                                        """)
        with gr.Column(scale=2):
            search = gr.Textbox(label='搜索结果')

        # 发送按钮 提交
        send.click(predict,
                   inputs=[
                       message,
                       chatbot,
                       # state
                   ],
                   outputs=[message, chatbot, state, search])


        # 输入框 回车
        message.submit(predict,
                       inputs=[
                           message,
                           chatbot
                       ],
                       outputs=[message, chatbot, state, search])

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    # server_port=8888,
    share=False,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=True,
)


    #query = "谁可以申请撤销监护人的监护资格?"
