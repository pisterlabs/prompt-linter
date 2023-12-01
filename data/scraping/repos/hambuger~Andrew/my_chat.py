import os
import sys
import uuid

import openai
from flask import request, render_template, Blueprint, session

from config.global_logger import logger as logging
from openai_util.chat import openai_chat_completions
from database_util.redis.redis_client import api_key_manager

my_chat_route = Blueprint('my_chat', __name__)

proxy_path = os.environ.get('PROXY_PATH', '')  # Get proxy path


def append_session(in_msg):
    try:
        # 获取session的值
        session_value = request.cookies.get("session")
        # 获取session的大小（字节）
        session_size = sys.getsizeof(session_value)
        # 如果session的大小超过4K token，将history里的内容取一半处理
        if session_size > 1000:
            # 获取history对象数组
            history = session["history"]
            # 计算数组的长度
            length = len(history)
            # 取数组的后一半
            half = history[length // 2:]
            # 用后一半替换原来的history
            session["history"] = half
            session.modified = True
        session['history'].append(in_msg)
        session.modified = True
        return "OK"
    except Exception:
        return "Something went wrong"


# 调用chatgpt对话
def chat(message_id, user_name, ip):
    openai.api_key = api_key_manager.get_openai_key()
    try:
        params = {
            "model": os.getenv('DEFAULT_CHAT_MODEL', 'gpt-3.5-turbo'),
            "messages": session['history']
        }
        return openai_chat_completions(params, message_id, "0", user_name, ip)
    except openai.error.RateLimitError:
        return "等会再聊，我太忙了"
    except Exception as e:
        logging.exception("chat error" + str(e))
        return None


# 定义一个路由，用于显示主页面
@my_chat_route.route('/')
def index():
    # 如果session中没有history，就创建一个空列表，用于保存用户的对话历史
    if 'history' not in session:
        session['history'] = []
    # 渲染一个模板，传入用户和chatgpt的默认头像和对话历史
    return render_template('index.html', user_avatar='user.png', chatgpt_avatar='chatgpt.png',
                           history=session['history'], proxy_path=proxy_path)


# 定义一个路由，用于处理用户的输入
@my_chat_route.route('/input', methods=['POST'])
def input_msg():
    # 获取用户的输入
    user_input = request.form.get('user_input')
    # 如果用户的输入不为空
    if not user_input:
        return "请输入"
    # 获取当前用户的ip，如果有代理，使用HTTP_X_FORWARDED_FOR头部，否则使用remote_addr属性
    user_ip = request.headers.get('HTTP_X_FORWARDED_FOR') or request.headers.get(
        'REMOTE-HOST') or request.remote_addr or '127.0.0.1'
    # 用户的ip和输入
    logging.info("user ip:" + user_ip + " user input:" + user_input)
    # 把用户的输入添加到对话历史中
    append_session({'role': 'user', 'content': user_input})
    session.modified = True
    # 用openai的聊天api生成一个回复
    message_id = 'my_chat-' + str(uuid.uuid4())
    chatgpt_reply = chat(message_id, os.getenv('MY_NAME', 'default'), user_ip)
    # 如果回复不为空
    if chatgpt_reply:
        # 把chatgpt的回复添加到对话历史中
        message = chatgpt_reply.get('choices')[0]['message']
        append_session(message)
        return message.get("content")
    else:
        return "等会再聊，我太忙了"
