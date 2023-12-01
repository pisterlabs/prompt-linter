# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b2_try_gradio.py
@Time: 2023-03-03 16:06
@Last_update: 2023-03-03 16:06
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import time
import gradio as gr
from lib.OpenaiBot import OpenaiBot
from lib.MongdbClient import MongodbClient

openai_bot = OpenaiBot()
mongo_client = MongodbClient()


def check_auth(username, passowrd):
    return mongo_client.check_user_exist(username, passowrd)


def ask_chatGPT(role, new_msg, state, request: gr.Request):
    """向chatGPT提问"""
    # 获取access_token
    access_token = request.request.cookies['access-token-unsecure']
    res_content = '对不起，服务器出错了，请稍后再试。'
    res = [(new_msg, res_content)]
    try:
        res_content = openai_bot.get_response(role, new_msg, state)
        res = [(new_msg, res_content)]
    except Exception as e:
        print(e)
    finally:
        state += res
        res = state

    # 更新history
    mongo_client.update_user_chat_history(access_token, new_msg, res_content)
    history, _, _ = mongo_client.get_user_chat_history(access_token)

    return res, state, history


def clean_question(question):
    """清除问题"""
    return ''


def clean_history(history, request: gr.Request):
    """清除历史记录"""
    access_token = request.request.cookies['access-token-unsecure']
    mongo_client.delete_user_chat_history(access_token)
    history, _, _ = mongo_client.get_user_chat_history(access_token)

    return history


def update_role(role, request: gr.Request):
    """更新角色"""
    access_token = request.request.cookies['access-token-unsecure']
    mongo_client.update_role(access_token, role)

    return role


if __name__ == '__main__':
    with gr.Blocks(title="尝试chatGPT对话", css="#maxheight {max-height: 390px} ") as demo:
        state = gr.State([])
        with gr.Column(variant='panel'):
            # title
            with gr.Row():
                gr.Markdown("## 尝试chatGPT对话")
            with gr.Row():
                # left part
                with gr.Column():
                    role_b = gr.Textbox(
                        label='请输入你设定的chatGPT的角色', lines=2,
                        value='你是ChatGPT，OpenAI训练的大规模语言模型，简明的回答用户的问题。')
                    question_b = gr.Textbox(
                        label='请输入你想要问的问题',
                        placeholder='输入你想提问的内容...',
                        lines=3
                    )
                    with gr.Row():
                        role_btn = gr.Button('更新角色')
                        greet_btn = gr.Button('提交', variant="primary")
                    with gr.Row():
                        clean_history_btn = gr.Button('清除历史记录')
                # right part
                with gr.Column():
                    # answer = gr.Textbox(
                    #     label='chatGPT的回答', lines=5, placeholder='等待回答...')
                    answer_b = gr.Chatbot(
                        label='chatGPT的问答', value=[(None, '请在这里提问')], elem_id='maxheight')
            with gr.Row():
                history_b = gr.TextArea(
                    label='历史记录', interactive=False)

        role_btn.click(fn=update_role, inputs=[role_b], outputs=[role_b])
        greet_btn.click(fn=ask_chatGPT, inputs=[role_b, question_b, state], outputs=[answer_b, state, history_b])
        greet_btn.click(fn=clean_question, inputs=[question_b], outputs=[question_b])
        clean_history_btn.click(fn=clean_history, inputs=[history_b], outputs=[history_b])

        def demo_load(request: gr.Request):
            """第一次进入demo时候运行的"""
            # 更新用户的access_token
            token_dict = demo.server_app.tokens
            access_token = request.request.cookies['access-token-unsecure']
            username = token_dict[access_token]
            mongo_client.update_user_access_token(username, access_token)

            # 获取用户的历史记录
            history_str, history_list, role = mongo_client.get_user_chat_history(access_token)

            return history_str, history_list[-10:], role, history_list[-10:]

        demo.load(demo_load, None, [history_b, state, role_b, answer_b])

    demo.launch(
        auth=check_auth, auth_message='请输入给定的用户名和密码', server_name='0.0.0.0', server_port=8081, show_api=False)
