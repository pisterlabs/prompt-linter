
import os
import sys
import json
from utils.redis_keys import WECHAT
from django_redis import get_redis_connection
from redis.client import Redis
from utils.redis_keys import WECHAT, Weather
import openai
import datetime
from core.celery import app
from celery_app.tasks import update_chatGPT_mode_time_remain

DEFAULT_REPLY_CONTENT = """查询内容格式有误,暂时只支持 [功能]:[参数1],[参数2].例如: 天气:广州市,揭阳市 \n
    切换到chatGPT模式,直接发送chatGPT:begin,退出chatGPT模式,直接发送chatGPT:end,若超过5分钟未问答,也将退出."""

TEXT_REPLY_XML_TEMPLATE = """
    <xml>
        <ToUserName><![CDATA[{to}]]></ToUserName>
        <FromUserName><![CDATA[{from_}]]></FromUserName>
        <CreateTime>12345678</CreateTime>
        <MsgType><![CDATA[text]]></MsgType>
        <Content><![CDATA[{content}]]></Content>
    </xml>
"""


WELCOME_CONETENT = """感谢关注!您可以输入[功能]:[参数1],[参数2]的形式,获取对应的信息.
例如: 天气:广州市,揭阳市 \n
如想切换到chatGPT模式,直接发送chatGPT:begin,退出chatGPT模式,直接发送chatGPT:end,chatGPT模式下若超过5分钟未问答,也将退出chatGPT模式
"""

CHATGPT_TIME = 60*5  # 超过改时间,将会自动退出CHATGPT模式

CHATGPT_MODE_ENTER_SUCCESS_REPLY = """进入GPT模式成功!"""
CHATGPT_MODE_EXIT_SUCCESS_REPLY = """退出成功!\n您可以输入[功能]:[参数1],[参数2]的形式,获取对应的信息.
例如: 天气:广州市,揭阳市 \n如想切换到chatGPT模式,直接发送chatGPT:begin,退出chatGPT模式,直接发送chatGPT:end,若超过5分钟未问答,也将退出"""
CHATGPT_ACTION_ERROR_REPLY="""参数有误,只支持:\nchatGPT:start (进入chatGPT模式) \n chatGPT:end (退出chatGPT模式)"""

CITY_ADCODE = dict()
with open(os.path.join(os.path.dirname(__file__), "city_adcode.json"), "r", encoding="utf-8") as f:
    CITY_ADCODE = json.load(f)

# 已经支持的功能
support_query_type = ["天气"]


def text_msg_handler(to: str, from_: str, content: str):
    try:
        if not is_in_chatGPT_mode(user_id=to):
            query_type, params = content.split(":", 1)
            if query_type == "chatGPT":
                action: str = params.split(",")[0]
                if action.strip() == "begin":
                    update_chatGPT_mode_time_remain(user_id=to)
                    content = CHATGPT_MODE_ENTER_SUCCESS_REPLY
                else:
                    content = "参数有误,输入chatGPT:begin可以进去chatGPT模式"
            elif query_type == "天气":
                city_list = params.split(",")
                conn: Redis = get_redis_connection()
                res = ""
                for city_cn_name in city_list:
                    city_adcode = CITY_ADCODE.get(city_cn_name)
                    if not city_adcode:
                        res = f"输入的查询参数:{params}有误"
                        break
                    info = conn.get(Weather.get_city_weather_key(city_adcode))
                    res += weather_info_format(city_name=city_cn_name,
                                            info=info)
                content = res
            else:
                content = f"{query_type}功能还未上线,目前已经支持的功能:{support_query_type}.也可以输入 chatGPT:begin 切换到chatGPT模式"
        else:
            try:
                query_type, params = content.split(":", 1)
                if query_type == "chatGPT":
                    action = params.split(",")[0]
                    if action.strip() == "end":
                        exit_chatGPT_mode(user_id=to)
                        content = CHATGPT_MODE_EXIT_SUCCESS_REPLY
            except:
                """
                    微信公众号的限制:因为微信公众号的回调超时是5秒,重试3次,这里跑够12秒.即第一次回调中调用openai API.在第三次回调10秒左右回复.12秒如果还无完整
                    答案,返回跳转到网页版本的链接.如果公众号有认证，可以通过客户接口来实现 #todo 
                """
                content = get_chatGPT_response(content=content)
                update_chatGPT_mode_time_remain.delay(user_id=to) # todo  async?
    except Exception as exc:
        import traceback
        print("处理微信文本消息异常", str(exc), traceback.format_exc())
        content = DEFAULT_REPLY_CONTENT

    return TEXT_REPLY_XML_TEMPLATE.format(
        to=to,
        from_=from_,
        content=content
    )


def weather_info_format(city_name, info):
    info = json.loads(info)
    forecasts = info.get("forecasts")
    for info in forecasts:
        updated = info.get("reporttime")
        res = """⬇⬇⬇[{city_cn_name}] 天气(更新于:{updated})⬇⬇⬇ \n""".format(
            city_cn_name=city_name, updated=updated)
        days_info = info.get("casts")
        for day_info in days_info:
            res += """[{date}]:\n白天:{dayweather}.温度:{daytemp}℃.风向:{daywind}风\n夜晚:{nightweather}.温度:{nighttemp}℃.风向:{nightwind}风\n""".format(
                date=day_info["date"],
                dayweather=day_info["dayweather"],
                daywind=day_info["daywind"],
                daytemp=day_info["daytemp"],
                nightweather=day_info["nightweather"],
                nighttemp=day_info["nighttemp"],
                nightwind=day_info["nightwind"],
            )
        return res

############################# chatGpt 相关 ######################################


def is_in_chatGPT_mode(user_id):
    conn: Redis = get_redis_connection()
    flag = conn.get(WECHAT.chatGpt_time_remain(wechat_id=user_id))
    return flag

def exit_chatGPT_mode(user_id):
    conn: Redis = get_redis_connection()
    conn.delete(WECHAT.chatGpt_time_remain(wechat_id=user_id))

def ignore(user_id):
    conn: Redis = get_redis_connection()
    return conn.get(WECHAT.chatGpt_result(wechat_id=user_id))


def get_chatGPT_response(content):
    openai.api_key = os.environ["OPENAPI_SECRET"]
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative,  \
            clever, and very friendly.\n\nHuman:{content}.\nAI:".format(content=content),
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["Human:", " AI:"]
    )
    res = ""
    if len(response["choices"]) > 1:
        res = "为您搜到以下答案:\n"
        for index, choice in enumerate(response["choices"]):
            res += f"回答[{index+1}]:\n{choice['text']}"
    else:
        res += response["choices"][0]['text']
    print("获取chatGPT结果 ---> ",res)
    return res
