# -*- coding: utf-8 -*-
import os
import time
import redis
import openai
import requests

import my_config
import SparkApi
import sensenova_service

from flask import Flask, request, abort, render_template
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.exceptions import (
    InvalidSignatureException,
    InvalidAppIdException,
)

# from bs4 import BeautifulSoup

# set token or get from environments
TOKEN = os.getenv('WECHAT_TOKEN', my_config.wechat_token)
AES_KEY = os.getenv('WECHAT_AES_KEY', my_config.wechat_aes_key)
APPID = os.getenv('WECHAT_APPID', my_config.wechat_app_id)
THINKING = 'THINKING'

# DEFAULT_MODEL = 'text-davinci-003'
DEFAULT_MODEL = 'gpt-3.5-turbo'
USER_MODEL_CACHE_PREFIX = 'user_model_'
CMD_LIST_MODEL = 'list model'
CMD_GET_MODEL = 'get model'
CMD_SET_MODEL = 'set model '

DEFAULT_SYS = 'xf'
USER_SYS_CACHE_PREFIX = 'user_sys_'
CMD_GET_SYS = 'get sys'
CMD_SET_SYS = 'set sys '

CMD_CALL_API = 'call api '

openai.api_key = my_config.key_openai


def chat(msg):
    sys = get_user_sys(msg.source)
    if sys == 'chatgpt':
        return chat_with_gpt3(msg)
    elif sys == 'writesonic':
        return chat_with_writesonic(msg)
    elif sys == 'xf':
        return chat_with_xf(msg)
    elif sys == 'sensenova':
        user_model = get_user_model(msg.source)
        return sensenova_service.ask_from_wechat(msg, user_model)
    else:
        return 'unknown sys'


def get_default_model(sys):
    if sys == 'sensenova':
        return sensenova_service.get_default_model()
    else:
        return DEFAULT_MODEL


def get_model_list(sys):
    if sys == 'sensenova':
        return sensenova_service.get_model_list()
    else:
        return [x.id for x in openai.Model.list().get('data')]


def chat_with_xf(msg):
    # https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
    # 用于配置大模型版本，默认“general/generalv2”
    # domain = "general"   # v1.5版本
    domain = "generalv2"  # v2.0版本
    # 云端环境的服务地址
    # Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
    Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
    SparkApi.answer = ""
    text = []
    jsoncon = {}
    jsoncon["role"] = "user"
    jsoncon["content"] = msg.content
    text.append(jsoncon)
    SparkApi.main(my_config.xf_appid, my_config.xf_api_key, my_config.xf_api_secret, Spark_url, domain, text)
    return SparkApi.answer


def chat_with_writesonic(msg):
    url = 'https://api.writesonic.com/v2/business/content/chatsonic?engine=premium'
    res = requests.post(url, json={
        'enable_google_results': 'false',
        'enable_memory': 'false',
        'input_text': msg.content,
    }, headers={
        "X-API-KEY": my_config.writesonic_api_key,
        'accept': 'application/json',
        'content-type': 'application/json',
    })
    print(res)
    return res.json()['message']


def chat_with_gpt3(msg):
    user_model = get_user_model(msg.source)
    if user_model in ['gpt-3.5-turbo']:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': msg.content}],
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=.9,
            user=msg.source,
        )
        message = response.choices[0].message.content
    else:
        response = openai.Completion.create(
            engine=user_model,
            prompt=msg.content,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
            user=msg.source,
        )
        message = response.choices[0].text
    print(response)

    return message


def get_redis():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    return r


def set_user_model(user, model):
    r = get_redis()
    if model == 'default':
        model = get_default_model(get_user_sys(user))
    r.set(USER_MODEL_CACHE_PREFIX + user, model)
    r.close()


def get_user_model(user):
    r = get_redis()
    user_model = r.get(USER_MODEL_CACHE_PREFIX + user)
    r.close()
    user_sys = get_user_sys(user)
    if user_model is None:
        user_model = get_default_model(user_sys)
    if user_model not in get_model_list(user_sys):
        user_model = get_default_model(user_sys)
    return user_model


def set_user_sys(user, sys):
    r = get_redis()
    if sys == 'default':
        sys = DEFAULT_SYS
    r.set(USER_SYS_CACHE_PREFIX + user, sys)
    r.close()


def get_user_sys(user):
    r = get_redis()
    user_sys = r.get(USER_SYS_CACHE_PREFIX + user)
    r.close()
    if user_sys is None:
        user_sys = DEFAULT_SYS
    return user_sys


def ask(msg, allow_cache=True):
    r = get_redis()
    key = 'chatgpt_answer_' + msg.content
    answer_cache = r.get(key)
    if answer_cache == THINKING and allow_cache:
        print('get THINKING')
        reply_content = 'LOOP'
        read_times = 10
        while read_times > 0:
            read_times -= 1
            time.sleep(1)
            answer_cache = r.get(key)
            if answer_cache is None:
                reply_content = 'loop result fail, try again'
                break
            elif answer_cache != THINKING:
                print('GOT')
                reply_content = answer_cache
                break
        reply_content += '[L]'
    elif answer_cache and allow_cache:
        print('answer from cache')
        reply_content = answer_cache
        reply_content += '[C]'
    else:
        print('call api')
        r.set(key, THINKING, 300)
        try:
            time_start = time.time()
            reply_content = chat(msg).lstrip('?？').strip()
            time_end = time.time()
            reply_content += '\n(time cost %.3f s)' % (time_end - time_start)
            r.set(key, reply_content, 28800)
        except openai.error.APIConnectionError as e:
            print(e)
            reply_content = 'call openapi fail, try again ' + e.__str__()
            r.delete(key)
        reply_content += '[I]'
    return reply_content


app = Flask(__name__)


@app.route('/')
def index():
    host = request.url_root
    return render_template('index.html', host=host)


@app.route('/wechat', methods=['GET', 'POST'])
def wechat():
    signature = request.args.get('signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    encrypt_type = request.args.get('encrypt_type', 'raw')
    msg_signature = request.args.get('msg_signature', '')
    try:
        check_signature(TOKEN, signature, timestamp, nonce)
    except InvalidSignatureException:
        abort(403)
    if request.method == 'GET':
        echo_str = request.args.get('echostr', '')
        return echo_str

    # POST request
    if encrypt_type == 'raw':
        # plaintext mode
        msg = parse_message(request.data)
        print(msg)
        if msg.type == 'text':
            reply_content = ''
            if msg.content == CMD_LIST_MODEL:
                user_sys = get_user_sys(msg.source)
                reply_content = '\n'.join(get_model_list(user_sys))
            elif msg.content.startswith(CMD_SET_MODEL):
                user_model = msg.content[len(CMD_SET_MODEL):]
                set_user_model(msg.source, user_model)
                reply_content = 'set user model success'
            elif msg.content == CMD_GET_MODEL:
                reply_content = get_user_model(msg.source)
            elif msg.content.startswith(CMD_SET_SYS):
                user_sys = msg.content[len(CMD_SET_SYS):]
                set_user_sys(msg.source, user_sys)
                reply_content = 'set user sys success'
            elif msg.content == CMD_GET_SYS:
                reply_content = get_user_sys(msg.source)
            elif msg.content.startswith(CMD_CALL_API):
                reply_content = ask(msg.content[len(CMD_CALL_API):], allow_cache=False)
            else:
                reply_content = ask(msg)
            print('ask {} response {}'.format(msg.content, reply_content))
            reply = create_reply(reply_content, msg)
        else:
            reply = create_reply('Sorry, can not handle this for now', msg)
        return reply.render()
    else:
        # encryption mode
        from wechatpy.crypto import WeChatCrypto

        crypto = WeChatCrypto(TOKEN, AES_KEY, APPID)
        try:
            msg = crypto.decrypt_message(request.data, msg_signature, timestamp, nonce)
        except (InvalidSignatureException, InvalidAppIdException):
            abort(403)
        else:
            msg = parse_message(msg)
            if msg.type == 'text':
                reply = create_reply(msg.content, msg)
            else:
                reply = create_reply('Sorry, can not handle this for now', msg)
            return crypto.encrypt_message(reply.render(), nonce, timestamp)


if __name__ == '__main__':
    app.run('127.0.0.1', 8081, debug=False)
