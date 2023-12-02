import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import time
import logging
import queue
import threading
import os

import glm_model
import model_manager
import state_manager
from model_manager import get_response_from_model
from state_manager import get_state, update_state, shared_output
from langchain.schema import HumanMessage, SystemMessage, AIMessage



output_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#定义数据库文件夹位置
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_files_dir = os.path.join(script_dir, 'Embedding_Files')

#界面视觉设定：
theme = gr.themes.Glass().set(
    body_background_fill='*primary_300',
    block_background_fill='*primary_100',
    block_border_width='3px',
    button_large_padding='*spacing_xs',
    button_large_radius='*radius_md',
    button_large_text_weight='500',
    button_small_radius='*radius_sm',
    button_small_text_size='*text_xs'
)

# 定义布局和组件
with gr.Blocks(theme=theme) as ui:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="聊天机器人", bubble_full_width=False, container=True, height=400, layout="panel") #avatar_images 元组[str |路径 |无，str |路径 |无] |没有默认值：无;用户和机器人的两个头像图像路径或 URL 的元组（按此顺序）。传递“无”，以
            msg = gr.Textbox(label="输入消息", placeholder="您好，我是一个专业数据库问答助手，请在这里输入问题……", lines=3)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        clear = gr.ClearButton([msg, chatbot], value="清除", min_width=88)
                        refresh = gr.Button("刷新", min_width=88)
                        send = gr.Button("发送", min_width=88)
                with gr.Column():
                    with gr.Row():
                        model_selector = gr.Dropdown(label="选择模型", choices=["OpenAI GPT-3.5", "Chat-GLM"],
                                                     value="OpenAI GPT-3.5")
                        temperature_UI = gr.Slider(label="温度", minimum=0, maximum=1, step=0.1)
            gr.Markdown("代理模块提示词控制台.")
            with gr.Tab("用户提问提示词"):
                template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2, value="你是一个健谈的问答助手，精通各个领域的知识，擅长理解用户的意图并思考以何种方案获得解决用户问题得答案")
                with gr.Row():
                    usr_clear = gr.Button("清空模板", min_width=88)
                    usr_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("分析器提示词"):
                analyzer_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2, value="你是一个聊天助手，擅长理解用户的意图并思考以何种方案获得解决用户的问题的方法并获取答案")
                with gr.Row():
                    analyzer_clear = gr.Button("清空模板", min_width=88)
                    analyzer_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("工具代理提示词"):
                tool_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    tool_clear = gr.Button("清空模板", min_width=88)
                    tool_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("搜索代理提示词"):
                search_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    search_clear = gr.Button("清空模板", min_width=88)
                    search_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("回答模板提示词"):
                answer_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    answer_clear = gr.Button("清空模板", min_width=88)
                    answer_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("临时提示词"):
                temporary_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    temporary_clear = gr.Button("清空模板", min_width=88)
                    temporary_Upload = gr.Button("保存模板", min_width=88)

        with gr.Column():
            example_image = gr.Image(label="示例图像")
            file_upload = gr.File(label="上传文件", file_count="multiple")
            FileExplorer = gr.FileExplorer(label="知识库文件管理器", root=embedding_files_dir)
            gr.Markdown("日志调试台.")
            with gr.Tab("Langchain日志"):
                log_output_box = gr.Textbox(label="Langchain日志", lines=16)
            with gr.Tab("代理反应"):
                agent_output_box = gr.Textbox(label="代理反应", lines=16)



    # 各模块的触发函数
    def threaded_model_call(llm_function, system_msg):
        model_manager.get_response_from_model(llm_function, system_msg)# 调用模型的函数


    def threaded_model_call_GLM(glm_key, message):
        glm_model.GLM_Streaming_response(message)



    def chat_function(model_choice, message, chat_history, temperature, template):
        global_state = get_state()
        global_state.module_template = template
        global_state.text_input = message  # 更新状态的值

        system_msg = [SystemMessage(content=global_state.module_template), HumanMessage(content=global_state.text_input)]

        chat_history.append((message, ""))
        yield "", chat_history

        # 根据选择的模型调用不同的处理函数
        if model_choice == 'OpenAI GPT-3.5':
            thread = threading.Thread(target=model_manager.get_response_from_model,
                                      args=(global_state.openai_model_info, system_msg))
        elif model_choice == 'Chat-GLM':
            thread = threading.Thread(target=glm_model.GLM_Streaming_response, args=(message,))
        else:
            raise ValueError("未知模型选择")

        thread.start()

        while True:
            token = state_manager.shared_output.get()
            if token is None:  # 检查结束信号
                logging.info("chat监测到结束信号")
                break
            logging.info(f"读取队列并界面刷新: {token}")
            chat_history[-1] = (message, chat_history[-1][1] + token)  # 更新消息
            yield "", chat_history

        clear_queue(shared_output)

        logging.info("流程结束，清空队列")
        return "", chat_history  # , global_state.agent_output_str, global_state.log_output_str, (
        #global_state.agent_output_str += f"这是代理的输出: {global_state.finish_answer}\n"
        #global_state.log_output_str += f"用户提问:{message},用户提示模板内容:{global_state.module_template},系统最终回答:{gpt_response.content}\n"




    def save_uploaded_file(uploaded_file):

        # 程序当前目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 指定的子目录，您可以根据需要更改这个名称
        storage_dir = os.path.join(script_dir, 'uploaded_files')

        # 如果存储目录不存在，则创建它
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        # 文件的存储路径
        file_path = os.path.join(storage_dir, uploaded_file.name)

        # 保存文件
        with open(file_path, "wb") as file:
            file.write(uploaded_file.read())

        return f"文件 {uploaded_file.name} 已上传"
    def clear_queue(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    #模板默认值

    # 绑定事件处理函数到按钮，按发送按钮触发输出
    send.click(chat_function, inputs=[model_selector, msg, chatbot, temperature_UI, template], outputs=[msg, chatbot])   #, agent_output_box, log_output_box, example_image])
    # 绑定函数到文本框和聊天机器人组件,按回车触发输出
    msg.submit(chat_function, inputs=[model_selector, msg, chatbot, temperature_UI, template], outputs=[msg, chatbot])  #, agent_output_box, log_output_box, example_image])
    # 绑定文件保存函数到文件上传组件
    file_upload.change(save_uploaded_file, inputs=[file_upload], outputs=[], show_progress=True, queue=True)
def start_UI(openai_model_info, glm_api_key, state_instance):
    global global_state
    global_state = state_instance
    global_state.openai_model_info = openai_model_info
    global_state.glm_model_info = glm_api_key
    ui.queue().launch(share=True, inbrowser=True)
