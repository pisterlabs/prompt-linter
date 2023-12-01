import json
import random
import time

import openai
import requests
import tiktoken

# 你的 api_key
import settings
from RedisUtil import RedisTool

chat_gpt_key = random.choice(settings.Config.chat_gpt_key.split(','))
# javastarboy 的腾讯云服务器函数服务，跳转硅谷区域代理
url = settings.Config.txProxyUrl
# 将 Key 传入 openai
openai.api_key = chat_gpt_key
# 模型 gpt-3.5-turbo-16k、gpt-3.5-turbo-0613
MODEL = "gpt-3.5-turbo-0613"

ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
"""
聊天信息（要记录历史信息，因为 AI 需要根据角色【user、system、assistant】上下文理解并做出合理反馈）
对话内容示例
messages = [
    {"role": "system", "content": "你是一个翻译家"},
    {"role": "user", "content": "将我发你的英文句子翻译成中文，你不需要理解内容的含义作出回答。"},
    {"role": "assistant", "content": "Draft an email or other piece of writing."}
]
"""
# 设置请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + chat_gpt_key,
    # 函数代理不想做鉴权，但又不想没校验，临时在头信息加了个校验
    "check": "check"
}


def clearMsg(FromUserName):
    print("已清除对话缓存")
    return dealUserSession(FromUserName, True)


"""
    调用 chatgpt 接口
"""


def completion(prompt, FromUserName):
    start_time = time.time()
    """
        API：https://api.openai.com/v1/chat/completions
        官方文档：https://platform.openai.com/docs/api-reference/chat
        :param FromUserName: 用户 id
        :param prompt: 入参文本框
        :return: 助手回答结果
    """
    # 设置请求体
    field = {
        "model": MODEL,
        "messages": prompt,
        "temperature": 0.0,
        "max_tokens": 500
    }
    # 发送 HTTP POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(field))
    print(f"=================》ChatGPT 实时交互完成，耗时 {time.time() - start_time} 秒。 返回信息为：{response.json()}", flush=True)

    # 解析响应结果
    if 'error' in response.json():
        error = response.json()['error']
        if 'code' in error and 'context_length_exceeded' == error['code']:
            resultMsg = '该模型的最大上下文长度是4096个令牌，请减少信息的长度或重设角色 (输入：stop) 创建新会话！。\n\n【' + error['message'] + "】"
    else:
        resultMsg = response.json()["choices"][0]["message"]["content"].strip()

    dealMsg(ROLE_ASSISTANT, resultMsg, '2', FromUserName)
    return resultMsg


def num_tokens_from_messages(infoMsg, model):
    """
        计算文本字符串中有多少个 token.
        非常长的对话更有可能收到不完整的回复。
        例如，一个长度为 4090 个 token 的 gpt-3.5-turbo 对话将在只回复了 6 个 token 后被截断。
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model.startswith("gpt-3.5-turbo"):  # 注意: 未来的模型可能会偏离这个规则
        num_tokens = 0
        for message in infoMsg:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # 如果有名字，角色将被省略
                    num_tokens += -1  # Role总是必需的，并且总是1个令牌
        num_tokens += 2  # 每个回复都用assistant启动
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def dealUserSession(FromUserName, clearType):
    """
    将 FromUserName 聊天记录存入 redis 中，以便支持多人会话，否则多人访问时，会有会话冲突的问题
    :param FromUserName: 微信用户 id
    :param clearType: 是否清空会话
    :return:
    """

    redis_tool = RedisTool().get_client()
    try:
        weChatToken = "WeChatGPT_" + FromUserName

        messages = redis_tool.get(weChatToken)
        if messages:
            messages = json.loads(messages)
            if clearType:
                redis_tool.delete(weChatToken)
                return "好的，您的会话 session 已清除，感谢使用!"
            else:
                # 存储消息【redis 取出来是 bytes 字节，需要转换一下】
                return messages
        elif clearType:
            return "好的，您的会话 session 已清除，感谢使用!"
        else:
            return None
    except Exception as e:
        print(f"An redis error occurred: {e}")
        raise ValueError("对不起，由于当前访问量过高，当前提问已被限制，请重新提问，谢谢~")
    finally:
        redis_tool.close()


def dealMsg(role, msg, msgRole, FromUserName):
    """
    :param role: 角色【system,user,assistant】
    :param msg: 聊天信息
    :param msgRole: 判断消息发送者【1-用户信息，2-助手信息】
    :param FromUserName: 用户 id
    :return:
    """
    weChatToken = "WeChatGPT_" + FromUserName
    messages = dealUserSession(FromUserName, False)
    redis_tool = RedisTool().get_client()
    try:
        if messages:
            messages.append({"role": role, "content": msg})
        elif messages is None:
            # 首次会话
            messages = [{"role": role, "content": msg}]

        redis_tool = RedisTool().get_client()
        # 默认一小时，每次更新数据都刷，如果一小时内都没有交互，默认删除 session
        redis_tool.setex(weChatToken, settings.Config.clearSessionTime, json.dumps(messages))
    except Exception as e:
        print(f"An redis error occurred: {e}")
        raise ValueError("对不起，由于当前访问量过高，当前提问已被限制，请重新提问，谢谢~")
    finally:
        redis_tool.close()

    # 如果是用户，做进一步处理后再请求 openai
    if msgRole == "1":
        # 计费：计算耗费的 token 数
        count = num_tokens_from_messages(messages, MODEL)
        print(f"{count} {msgRole} prompt tokens counted")
        if count > 4096:
            raise ValueError("请求上下文已超过 4096 令牌数，请重设角色 (输入：stop) 创建新会话！")

        """
            如果列表长度大于 6，删除多余的数据，只保留第一条以及后4 或 5条数据（带上下文请求 gpt）
            第一条：role = system
            后三条：历史上下文保留三条数据（含当前最新消息）
        """
        if len(messages) > 6:
            if messages[-1]["role"] == ROLE_USER:
                # 主要针对"继续写"场景，最后一条为 user 一定是用户问了新问题
                print([messages[0]] + messages[-4:])
            else:
                # 第一条 + 后五条记录（最后一条非 role，那一定是 assistant）
                print([messages[0]] + messages[-5:])
    # 历史消息
    return messages
