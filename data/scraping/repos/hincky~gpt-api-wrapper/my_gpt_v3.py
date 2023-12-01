import datetime
import json
import logging

import tiktoken
from flask import Flask, request, make_response, Response, stream_with_context
import random
import time
import openai

auth_token = "your auth token"


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


app = Flask(__name__)

# API key列表文件名
api_keys_file = 'api_keys'

# 无效API key列表文件名
invalid_api_keys_file = 'invalid_api_keys'

# API key列表
api_keys = []

# 无效API key列表
invalid_api_keys = []

# 从文件中读取API key列表
with open(api_keys_file, 'r') as f:
    api_keys = [line.strip() for line in f]

messages_cache = {}


@app.route('/chat/gpt-35', methods=['POST'])
def gpt35_chat():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    user_id = request.args.get('user_id', None, str)
    system_message = request.args.get('systemMessage',
                                      f"You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: {datetime.date.today()}",
                                      str)
    content = request.data.decode("UTF-8")
    print(current_time(), "=============user_id： ", 'None' if user_id is None else user_id)
    print(current_time(), "=============输入的问题： ", content)

    global messages_cache
    if user_id is None or user_id == '':
        messages = [{
            "role": "system",
            "content": system_message
        }]
    elif user_id in messages_cache.keys():
        messages = messages_cache[user_id]
    else:
        messages = [{
            "role": "system",
            "content": system_message
        }]

    messages.append({"role": "user", "content": content})

    while num_tokens_from_messages(messages) > 2000 and len(messages) > 3:
        del messages[1]

    print(current_time(), "=============请求的messages长度： ", len(messages))
    message, totalTokens, spend, failTimes = get_result_by_gpt(messages, "gpt-3.5-turbo")
    messages.append(message)
    messages_cache[user_id] = messages

    result = {
        "response": message.content,
        "spend": spend,
        "totalTokens": totalTokens,
        "failTimes": failTimes,
        "historyLength": len(messages)
    }

    return make_response(result, 200)


@app.route('/chat/gpt-4', methods=['POST'])
def gpt4_chat():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    user_id = request.args.get('user_id', None, str)
    system_message = request.args.get('systemMessage',
                                      f"You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: {datetime.date.today()}",
                                      str)
    content = request.data.decode("UTF-8")
    print(current_time(), "=============user_id： ", 'None' if user_id is None else user_id)
    print(current_time(), "=============输入的问题： ", content)

    global messages_cache
    if user_id is None or user_id == '':
        messages = [{
            "role": "system",
            "content": system_message
        }]
    elif user_id in messages_cache.keys():
        messages = messages_cache[user_id]
    else:
        messages = [{
            "role": "system",
            "content": system_message
        }]

    messages.append({"role": "user", "content": content})

    while num_tokens_from_messages(messages) > 2000 and len(messages) > 3:
        del messages[1]

    print(current_time(), "=============请求的messages长度： ", len(messages))
    message, totalTokens, spend, failTimes = get_result_by_gpt(messages, "gpt-4-0314")
    messages.append(message)
    messages_cache[user_id] = messages

    result = {
        "response": message.content,
        "spend": spend,
        "totalTokens": totalTokens,
        "failTimes": failTimes,
        "historyLength": len(messages)
    }

    return make_response(result, 200)


@app.route('/chat/stream', methods=['POST'])
def stream_chat():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    user_id = request.args.get('user_id', None, str)
    model = request.args.get('model', "gpt-3.5-turbo-0301", str)
    system_message = request.args.get('systemMessage',
                                      f"You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: {datetime.date.today()}",
                                      str)
    content = request.data.decode("UTF-8")
    print(current_time(), "=============user_id： ", 'None' if user_id is None else user_id)
    print(current_time(), "=============输入的问题： ", content)

    global messages_cache
    if user_id is None or user_id == '':
        messages = [{
            "role": "system",
            "content": system_message
        }]
    elif user_id in messages_cache.keys():
        messages = messages_cache[user_id]
    else:
        messages = [{
            "role": "system",
            "content": system_message
        }]

    messages.append({"role": "user", "content": content})

    while (num_tokens_from_messages(messages) > 2000 and len(messages) > 3) or len(messages) > 10:
        del messages[1]

    print(current_time(), "=============请求的messages长度： ", len(messages))
    def generate_numbers():
        # record the time before the request is sent
        start_time = time.time()

        # send a ChatCompletion request to count to 100
        response = None
        flag = True
        i = 1
        while flag and i <= 10:
            api_key = random.choice(api_keys)
            try:
                print(current_time(), "第" + str(i) + f"次请求openai的{model}接口")

                response = openai.ChatCompletion.create(
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.8,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=1,
                    stream=True
                )

                flag = False
            except openai.error.OpenAIError as e:
                i = i + 1
                print(current_time(), "openai服务发生错误： " + str(e.user_message))

        # iterate through the stream of events
        fullMessage = []
        result = {}
        for chunk in response:
            # yield f'{chunk}\n'
            chunk_message = chunk['choices'][0]['delta']
            if chunk_message.get('role') is not None:
                continue
            if chunk_message.get('content') is not None:
                fullMessage.append(chunk_message)
            if chunk['choices'][0].get('finish_reason') == 'stop':
                # 将耗时和消耗的总token返回
                spend = time.time() - start_time
                totalTokens = num_tokens_from_messages(messages) + num_tokens_from_messages(fullMessage)
                result = {
                    "stop": "true",
                    "response": ''.join([m.get('content', '') for m in fullMessage]),
                    "spend": spend,
                    "totalTokens": totalTokens,
                    "failTimes": i - 1,
                    "historyLength": len(messages)
                }
                chunk_message = json.dumps(result)
            yield f"{chunk_message}"

        print(f"========gpt api完整回复内容: {str(result.get('response'))}")
        print(current_time(), "total_tokens: " + str(result.get('totalTokens')))
        print(current_time(), "耗时: " + str(result.get('spend')) + "秒")

        messages.append({
            "role": "assistant",
            "content": str(result.get('response'))
        })
        messages_cache[user_id] = messages

    return Response(generate_numbers(), content_type='text/event-stream')

def check_request_and_log():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    content = request.data.decode("UTF-8")
    print(current_time(), "=============输入的prompt： ")
    print(current_time(), content)
    print(current_time(), "=============prompt结束")
    encoding = tiktoken.get_encoding("gpt2")
    input_length = len(encoding.encode(content))
    if input_length > 3000:
        print(current_time(), "=============输入的prompt计算的token长度： ")
        print(current_time(), input_length)
        return make_response("prompt参数不能超过3000 tokens，目前是： " + str(input_length) + "，请修改后重试", 400)
    return content


def get_result_by_gpt(messages, model="gpt-3.5-turbo"):
    completion_start_time = time.time() * 1000
    completion = None
    flag = True
    i = 1
    while flag and i <= 10:
        api_key = random.choice(api_keys)
        try:
            print(current_time(), "第" + str(i) + f"次请求openai的{model}接口")

            completion = openai.ChatCompletion.create(
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.9,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0
            )

            flag = False
        except openai.error.OpenAIError as e:
            i = i + 1
            print(current_time(), "openai服务发生错误： " + str(e.user_message))
    completion_end_time = time.time() * 1000
    if completion is None:
        return make_response("openai服务无法响应", 500)
    print(current_time(), "=============gpt api响应结果： ")
    print(current_time(), "response: " + str(completion.choices[0].message.content))
    print(current_time(), "total_tokens: " + str(completion["usage"]["total_tokens"]))
    print(current_time(), "耗时: " + str(round(completion_end_time - completion_start_time)))

    return completion.choices[0].message, completion["usage"]["total_tokens"], round(
        completion_end_time - completion_start_time), i


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


@app.route('/', methods=['GET'])
def health_check():
    return "success", 200


log = logging.getLogger('werkzeug')
log.setLevel(logging.WARN)

if __name__ == '__main__':
    # 启动Web服务
    app.run(host='0.0.0.0', port=5070)
