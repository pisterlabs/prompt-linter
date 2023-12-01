import logging

import tiktoken
from flask import Flask, jsonify, request, make_response, Response
import requests
import random
import time
import openai

auth_token = "your auth token"


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


cache = {
    "model": "gpt3.5"
}

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


@app.route('/change_model', methods=['POST'])
def change_model():
    global cache
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    if request.args.get("model", str) is not None:
        cache["model"] = request.args.get("model", str)
    return make_response("ok", 200)


@app.route('/chat/gpt-35', methods=['POST'])
def gpt35_chat():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    user_id = request.args.get('user_id', None, str)
    content = request.data.decode("UTF-8")
    print(current_time(), "=============user_id： ",  'None' if user_id is None else user_id)
    print(current_time(), "=============输入的问题： ", content)

    global messages_cache
    if user_id is None or user_id == '':
        messages = []
    elif user_id in cache.keys():
        messages = cache[user_id]
    else:
        messages = []

    messages.append({"role": "user", "content": content})

    encoding = tiktoken.get_encoding("gpt2")
    while len(encoding.encode(str(messages))) > 3000:
        messages.pop(0)

    print(current_time(), "=============请求的messages长度： ", len(messages))
    message, totalTokens, spend, failTimes = get_result_by_gpt(messages, "gpt-3.5-turbo")
    messages.append(message)
    cache[user_id] = messages

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
    content = request.data.decode("UTF-8")
    print(current_time(), "=============user_id： ",  'None' if user_id is None else user_id)
    print(current_time(), "=============输入的问题： ", content)

    global messages_cache
    if user_id is None or user_id == '':
        messages = []
    elif user_id in cache.keys():
        messages = cache[user_id]
    else:
        messages = []

    messages.append({"role": "user", "content": content})

    encoding = tiktoken.get_encoding("gpt2")
    while len(encoding.encode(str(messages))) > 7000:
        messages.pop(0)

    print(current_time(), "=============请求的messages长度： ", len(messages))
    message, totalTokens, spend, failTimes = get_result_by_gpt(messages, "gpt-4-0314")
    messages.append(message)
    cache[user_id] = messages

    result = {
        "response": message.content,
        "spend": spend,
        "totalTokens": totalTokens,
        "failTimes": failTimes,
        "historyLength": len(messages)
    }

    return make_response(result, 200)


@app.route('/gpt-3/davinci', methods=['POST'])
def gpt3_test():
    content = check_request_and_log()
    if isinstance(content, Response):
        return content
    global cache

    if cache is not None and cache["model"] == "gpt3.5":
        result = get_result_by_gpt35(content)
    else:
        result = get_result_by_gpt3(content)

    if isinstance(result, Response):
        return result

    return make_response(result, 200)


@app.route('/gpt-3.5/turbo', methods=['POST'])
def gpt35_turbo():
    content = check_request_and_log()
    if isinstance(content, Response):
        return content

    result = get_result_by_gpt35(content)
    if isinstance(result, Response):
        return result

    return make_response(result, 200)


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
                temperature=0.5,
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


def get_result_by_gpt35(content):
    completion_start_time = time.time() * 1000
    completion = None
    flag = True
    i = 1
    while flag and i <= 10:
        api_key = random.choice(api_keys)
        try:
            print(current_time(), "第" + str(i) + "次请求openai的gpt-3.5接口")

            completion = openai.ChatCompletion.create(
                api_key=api_key,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": content}
                ],
                max_tokens=1000,
                temperature=0,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0
            )

            flag = False
        except openai.error.OpenAIError as e:
            i = i + 1
            # if isinstance(e, openai.error.RateLimitError) or isinstance(e, openai.error.AuthenticationError):
            # 删除key
            # print(f"无效的key， {api_key}")
            # del_invalid_key(api_key)
            print(current_time(), "openai服务发生错误： " + str(e.user_message))
    completion_end_time = time.time() * 1000
    if completion is None:
        return make_response("openai服务无法响应", 500)
    print(current_time(), "=============gpt api响应结果： ")
    print(current_time(), "response: " + str(completion.choices[0].message.content))
    print(current_time(), "total_tokens: " + str(completion["usage"]["total_tokens"]))
    print(current_time(), "耗时: " + str(round(completion_end_time - completion_start_time)))

    result = {
        "response": completion.choices[0].message.content,
        "spend": round(completion_end_time - completion_start_time),
        "totalTokens": completion["usage"]["total_tokens"],
        "failTimes": i - 1
    }
    return result


def del_invalid_key(api_key):
    api_keys.remove(api_key)
    invalid_api_keys.append(api_key)
    print(current_time(), f"剩余有效的api_key个数 {len(api_keys)}")
    with open(api_keys_file, 'w') as f:
        for key in api_keys:
            f.write(key + '\n')
    with open(invalid_api_keys_file, 'a') as f:
        f.write(api_key + '\n')


def get_result_by_gpt3(content):
    completion_start_time = time.time() * 1000
    response = None
    flag = True
    i = 1
    while flag and i <= 100:
        try:
            print(current_time(), "第" + str(i) + "次请求openai的gpt-3接口")
            response = openai.Completion.create(
                api_key=random.choice(api_keys),
                model="text-davinci-003",
                prompt=content,
                max_tokens=1000,
                temperature=0,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0
            )
            flag = False
        except openai.error.OpenAIError as e:
            i = i + 1
            print(current_time(), "openai服务发生错误： " + str(e.user_message))
    completion_end_time = time.time() * 1000
    if response is None:
        return make_response("openai服务无法响应", 500)
    print(current_time(), "=============gpt api响应结果： ")
    print(current_time(), "response: " + str(response["choices"][0]["text"]))
    print(current_time(), "total_tokens: " + str(response["usage"]["total_tokens"]))
    print(current_time(), "耗时: " + str(round(completion_end_time - completion_start_time)))

    result = {
        "response": response["choices"][0]["text"],
        "spend": round(completion_end_time - completion_start_time),
        "totalTokens": response["usage"]["total_tokens"],
        "failTimes": i - 1
    }
    return result


# 动态增加API key的接口
@app.route('/add_api_key', methods=['POST'])
def add_api_key():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    keys = request.form.get('keys', '', str)
    keys = keys.split("\n")
    global api_keys
    if keys and len(keys) > 0:
        api_keys = api_keys + keys
        for api_key in keys:
            with open(api_keys_file, 'a') as f:
                f.write(api_key + '\n')
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'message': 'api_key missing'})


# 获取key的余额
def get_usd_available(api_key):
    url = 'https://api.openai.com/dashboard/billing/credit_grants'
    headers = {'Authorization': f'Bearer {api_key}'}
    # params = {'param1': 'value1', 'param2': 'value2'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # 处理返回的JSON数据
        total_available = data['total_available']
        if total_available > 0:
            print(f"剩余{total_available}美元")
            return True
        else:
            print(f"{api_key} 免费额度已用完")
            return False
    elif response.status_code == 401:
        print('key无效：', api_key)
        return False
    else:
        print('请求失败：', response.status_code)
        return True


# 定期检查和更新API key列表
@app.route('/check_api_keys', methods=['POST'])
def check_api_keys_api():
    if request.args.get("auth_token", '', str) is None or request.args.get("auth_token", '', str) != auth_token:
        return make_response("auth_token认证失败", 401)
    check_api_keys()
    return make_response(f"剩余有效的api_key个数 {len(api_keys)}", 200)


def check_api_keys():
    # while True:
    print(current_time(), f"验证api_keys是否都有效，个数 {len(api_keys)}")
    i = 1
    for api_key in api_keys:
        print(current_time(), f"验证第{i}个api_key: {api_key}")
        result = get_usd_available(api_key)
        i += 1
        if not result:
            print(current_time(), f"key {api_key} 无效，从缓存中去除")
            api_keys.remove(api_key)
            invalid_api_keys.append(api_key)
            print(current_time(), f"剩余有效的api_key个数 {len(api_keys)}")
            with open(api_keys_file, 'w') as f:
                for key in api_keys:
                    f.write(key + '\n')
                with open(invalid_api_keys_file, 'a') as f:
                    f.write(api_key + '\n')
        # time.sleep(3)


@app.route('/', methods=['GET'])
def health_check():
    return "success", 200


log = logging.getLogger('werkzeug')
log.setLevel(logging.WARN)

# 启动检查API key的线程
# t = threading.Thread(target=check_api_keys)
# t.daemon = True
# t.start()
# check_api_keys()

if __name__ == '__main__':
    # 启动Web服务
    app.run(host='0.0.0.0', port=5050)
