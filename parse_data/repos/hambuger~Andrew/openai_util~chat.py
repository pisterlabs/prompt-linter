import json
from config.global_logger import logger
import os

import openai
from flask import request, Blueprint, Response
from openai import OpenAIError
from openai_util.prompt import get_hg_prompt
from memory.remember import insert_history
from openai_util.msg_deal import generate_messages_v3
from database_util.redis.redis_client import api_key_manager
from openai_util.embedding import get_embedding
from concurrent.futures import ThreadPoolExecutor

chat_route = Blueprint('chat', __name__)

executor = ThreadPoolExecutor(10)


# @chat_route.route('/v1/models', methods=['GET'])
def model():
    auth_info = request.headers.get('Authorization')
    set_req_api_key(auth_info)
    try:
        return openai.Model.list()
    except OpenAIError as e:
        logger.exception(e)
        return e.message


@chat_route.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions_for_web():
    try:
        # Get incoming parameters
        (params, message_id, parent_id, userName, ip) = deal_request_param()
        return openai_chat_completions(params, message_id, parent_id, userName, ip)
    except OpenAIError as e:
        logger.exception(e.message)
        return e.message
    except Exception as e:
        logger.exception("error: {}".format(e))
        return "Unknown error, please contact hamburger"


def openai_chat_completions(params, message_id, parent_id, user_name, ip):
    stream_flag = params.get('stream')
    function_call = params.get('function_call')
    # save chat history
    add_message_record(params, message_id, parent_id, user_name, ip)
    # Pass in parameters when calling the API
    response = openai.ChatCompletion.create(**params)
    # Handle stream returns
    return deal_stream_response(function_call, stream_flag, response, message_id, user_name, ip)


# Set the api_key of the request
def set_req_api_key(auth_info):
    need_user_api_key = os.getenv('NEED_USER_API_KEY', False) == 'True'
    api_key = None
    if auth_info:
        api_key = auth_info.replace('Bearer ', '')
        if not need_user_api_key and (not api_key or not api_key.startswith('sk-')):
            api_key = api_key_manager.get_openai_key()
    elif not need_user_api_key:
        api_key = api_key_manager.get_openai_key()
    openai.api_key = api_key


# Insert user's chat history
def add_message_record(params, message_id, parent_id, user_name, ip):
    messages = params.get('messages')
    content = messages[-1].get("content")
    content_vector = get_embedding(content)
    params['messages'] = generate_messages_v3(content, content_vector, user_name, ip, messages)
    logger.debug('messages: {}'.format(params['messages']))
    # Asynchronous processing to save chat records
    executor.submit(insert_history, message_id, parent_id, ip, user_name, user_name, content, content_vector, 0, [])


# Insert the return record of GPT
def add_response_record(all_contents, bot_msg_ids, parent_id, ip, user_name):
    content_vector = get_embedding(''.join(all_contents))
    insert_history(bot_msg_ids[0], parent_id, ip, user_name, 'gpt-3.5', ''.join(all_contents), content_vector, 0, [])


def insert_ai_response_record(content, msg_id, parent_id, ip, user_name):
    content_vector = get_embedding(content)
    insert_history(msg_id, parent_id, ip, user_name, 'gpt-3.5', content, content_vector, 0, [])


# Handle stream returns
def deal_stream_response(function_call, stream_flag, response, parent_id, user_name, ip):
    all_contents = []
    botMsgIds = []

    def stream_response():
        for chunk in response:
            content = chunk['choices'][0]['delta'].get('content', '')
            all_contents.append(content)
            botMsgIds.append(chunk['id'])
            yield 'data: ' + json.dumps(chunk) + '\n\n'
        executor.submit(add_response_record, all_contents, botMsgIds, parent_id, ip, user_name)

    if stream_flag:
        return Response(stream_response(), mimetype='application/octet-stream', content_type='application/json')
    else:
        if not function_call:
            executor.submit(insert_ai_response_record, response['choices'][0]['message']['content'], response['id'],
                            parent_id,
                            ip, user_name)
        return response


# Handle request parameters
def deal_request_param():
    # Get api_key to determine whether you need to use the system api_key
    need_user_api_key = os.getenv('NEED_USER_API_KEY', False) == 'True'
    auth_info = request.headers.get('Authorization')
    api_key = None
    if auth_info:
        api_key = auth_info.replace('Bearer ', '')
        if not need_user_api_key and (not api_key or not api_key.startswith('sk-')):
            api_key = api_key_manager.get_openai_key()
    elif not need_user_api_key:
        api_key = api_key_manager.get_openai_key()
    openai.api_key = api_key
    # Get interface parameters
    req_model = request.json.get('model') or os.getenv('DEFAULT_CHAT_MODEL', 'gpt-3.5-turbo')
    messages_req = request.json.get('messages')
    message_id = messages_req[-1].get('id')
    parent_id = None
    if len(messages_req) > 1:
        parent_id = messages_req[-2].get('id')
    # Handle special information incoming from the front end
    messages = [{k: v for k, v in msg.items() if k in ['role', 'content', 'function_call']} for msg in messages_req]
    # Store the names of all fields that need to be checked
    fields_to_check = ['logit_bias', 'temperature', 'top_p', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
                       'frequency_penalty', 'user', 'functions']
    # Prepare the necessary parameters for the API call
    params = {
        'model': req_model,
        'messages': messages,
    }
    # Iterate through all the fields that need to be checked
    for field in fields_to_check:
        # Get field values from request.json using .get() method
        field_value = request.json.get(field)
        # If the obtained field value is not None, add it to params
        if field_value is not None:
            params[field] = field_value
    client_ip = request.headers.get('X-Forwarded-For', default=request.remote_addr)
    ip = request.json.get('ip') or client_ip
    userName = request.json.get('user_name') or 'default'
    logger.debug('ip: {}'.format(ip))
    if not message_id:
        logger.warning('params error: {}'.format(params))
        return None, None, None, None, None
    return params, message_id, parent_id, userName, ip


def hgchat(messages):
    openai.api_key = api_key_manager.get_gh_chat_model_key()
    try:
        response = openai.Completion.create(
            model=os.getenv("MY_CHAT_MODEL"),
            prompt=get_hg_prompt(messages),
            max_tokens=500,
            temperature=0.4,
            stop=["end"]
        )
        return response
    except Exception as e:
        logger.debug("hgchat error" + str(e))
        return None
