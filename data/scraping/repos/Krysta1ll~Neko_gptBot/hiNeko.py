import json
import os
import traceback
import uuid
from copy import deepcopy
from flask import request, Flask
import openai
import requests
from weather_neko import getweather
from opBot import opBot

from text_to_image import text_to_image

with open("config.json", "r",
          encoding='utf-8') as jsonfile:
    config_data = json.load(jsonfile)
    qq_no = config_data['qq_bot']['qq_no']

session_config = {
    'msg': [
        {"role": "system", "content": config_data['chatgpt']['preset']}
    ]
}

sessions = {}
current_key_index = 0

openai.api_base = "https://chat-gpt.aurorax.cloud/v1"

# 创建flask服务
server = Flask(__name__)

# ban list
banList = ["毛泽", "泽东", "习近", "共产党", "政治", "疫情", "历史", "极权", "主义", "移民", "连任",
           "皇帝", "洗脑"]

# 和风天气api
we_url = "https://devapi.qweather.com/v7/weather/3d"
we_key = "4b2e29110cd94ccb8cf172175aac140e"
we_location = ""


# 测试
@server.route('/', methods=["GET"])
def index():
    return f"你好，世界!<br/>"


# 获取账号余额接口
@server.route('/credit_summary', methods=["GET"])
def credit_summary():
    return get_credit_summary()


# qq消息上报接口，qq机器人监听到的消息内容将被上报到这里
@server.route('/', methods=["POST"])
def get_message():
    if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
        uid = request.get_json().get('sender').get('user_id')  # 获取信息发送者的 QQ号码
        message = request.get_json().get('raw_message')  # 获取原始信息
        sender = request.get_json().get('sender')  # 消息发送者的资料
        print("收到私聊消息：")
        print(message)
        # 下面你可以执行更多逻辑，这里只演示与ChatGPT对话
        if message.strip().startswith('生成图像'):
            message = str(message).replace('生成图像', '')
            msg_text = chat(message, 'P' + str(uid))  # 将消息转发给ChatGPT处理
            # 将ChatGPT的描述转换为图画
            print('开始生成图像')
            pic_path = get_openai_image(msg_text)
            send_private_message_image(uid, pic_path, msg_text)
        elif message.strip().startswith('直接生成图像'):
            message = str(message).replace('直接生成图像', '')
            print('开始直接生成图像')
            pic_path = get_openai_image(message)
            send_private_message_image(uid, pic_path, '')
        else:
            msg_text = chat(message, 'P' + str(uid))  # 将消息转发给ChatGPT处理
            send_private_message(uid, msg_text)  # 将消息返回的内容发送给用户

    if request.get_json().get('message_type') == 'group':  # 如果是群消息
        gid = request.get_json().get('group_id')  # 群号
        uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
        message = request.get_json().get('raw_message')  # 获取原始信息
        # 判断当被@时才回答
        if str("[CQ:at,qq=%s]" % qq_no) in message:
            sender = request.get_json().get('sender')  # 消息发送者的资料
            print("收到群聊消息：")
            print(message)
            message = str(message).replace(str("[CQ:at,qq=%s]" % qq_no), '')
            if message.strip().startswith('生成图像'):
                message = str(message).replace('生成图像', '')
                msg_text = chat(message, 'G' + str(gid))  # 将消息转发给ChatGPT处理
                # 将ChatGPT的描述转换为图画
                print('开始生成图像')
                pic_path = get_openai_image(msg_text)
                send_group_message_image(gid, pic_path, uid, msg_text)
            elif message.strip().startswith('直接生成图像'):
                message = str(message).replace('直接生成图像', '')
                print('开始直接生成图像')
                pic_path = get_openai_image(message)
                send_group_message_image(gid, pic_path, uid, '')

            else:
                # 下面你可以执行更多逻辑，这里只演示与ChatGPT对话
                msg_text = chat(message, 'G' + str(gid))  # 将消息转发给ChatGPT处理
                send_group_message(gid, msg_text, uid)  # 将消息转发到群里

    if request.get_json().get('post_type') == 'request':  # 收到请求消息
        print("收到请求消息")
        request_type = request.get_json().get('request_type')  # group
        uid = request.get_json().get('user_id')
        flag = request.get_json().get('flag')
        comment = request.get_json().get('comment')
        print("配置文件 auto_confirm:" + str(config_data['qq_bot']['auto_confirm']) + " admin_qq: " + str(
            config_data['qq_bot']['admin_qq']))
        if request_type == "friend":
            print("收到加好友申请")
            print("QQ：", uid)
            print("验证信息", comment)
            # 如果配置文件里auto_confirm为 TRUE，则自动通过
            if config_data['qq_bot']['auto_confirm']:
                set_friend_add_request(flag, "true")
            else:
                if str(uid) == config_data['qq_bot']['admin_qq']:  # 否则只有管理员的好友请求会通过
                    print("管理员加好友请求，通过")
                    set_friend_add_request(flag, "true")
        if request_type == "group":
            print("收到群请求")
            sub_type = request.get_json().get('sub_type')  # 两种，一种的加群(当机器人为管理员的情况下)，一种是邀请入群
            gid = request.get_json().get('group_id')
            if sub_type == "add":
                # 如果机器人是管理员，会收到这种请求，请自行处理
                print("收到加群申请，不进行处理")
            elif sub_type == "invite":
                print("收到邀请入群申请")
                print("群号：", gid)
                # 如果配置文件里auto_confirm为 TRUE，则自动通过
                if config_data['qq_bot']['auto_confirm']:
                    set_group_invite_request(flag, "true")
                else:
                    if str(uid) == config_data['qq_bot']['admin_qq']:  # 否则只有管理员的拉群请求会通过
                        set_group_invite_request(flag, "true")
    return "ok"


# 判断是否有违规内容
def testpolicy(message):
    for i in range(len(banList)):
        if message.__contains__(banList[i]):
            return False


# 测试接口，可以用来测试与ChatGPT的交互是否正常，用来排查问题
@server.route('/chat', methods=['post'])
def chatapi():
    request_json = request.get_data()
    if request_json is None or request_json == "" or request_json == {}:
        resu = {'code': 1, 'msg': '请求内容不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(request_json)
    if data.get('id') is None or data['id'] == "":
        resu = {'code': 1, 'msg': '会话id不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    print(data)
    try:
        msg = chat(data['msg'], data['id'])
        if '查询余额' == data['msg'].strip():
            msg = msg.replace('\n', '<br/>')
        resu = {'code': 0, 'data': msg, 'id': data['id']}
        return json.dumps(resu, ensure_ascii=False)
    except Exception as error:
        print("接口报错")
        resu = {'code': 1, 'msg': '请求异常: ' + str(error)}
        return json.dumps(resu, ensure_ascii=False)


# 重置会话接口
@server.route('/reset_chat', methods=['post'])
def reset_chat():
    request_json = request.get_data()
    if request_json is None or request_json == "" or request_json == {}:
        resu = {'code': 1, 'msg': '请求内容不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(request_json)
    if data['id'] is None or data['id'] == "":
        resu = {'code': 1, 'msg': '会话id不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    # 获得对话session
    session = get_chat_session(data['id'])
    # 清除对话内容但保留人设
    del session['msg'][1:len(session['msg'])]
    resu = {'code': 0, 'msg': '重置成功'}
    return json.dumps(resu, ensure_ascii=False)


# 与ChatGPT交互的方法
def chat(msg, sessionid):
    try:
        if msg.strip() == '':
            return '喵喵！我是猫娘Scarlet，是一个集成了GPT3.5的猫娘哟，有什么问题都可以问我喵！\n如果想使用时间回溯技能，请回复“重置会话”'
        # 获得对话session
        session = get_chat_session(sessionid)

        # 关键词屏蔽,交互前判断
        if testpolicy(msg) == False:
            print("存在敏感内容_From User")
            return "竟然有人提及了不能说的话题qwq，猫娘不喜欢你了喵！💔"

        # 實現其他功能
        if 'yao' == msg.strip():
            return "張玉瑤是中南大學acm冠軍，並且高數滿分"
        if msg.__contains__("原神"):
            return opBot()

        # 天气预报
        if msg.__contains__("天气"):
            if msg.__contains__("今天天气"):
                print(msg)
                city = msg.strip().split('今')[0]
                print(city)
                return getweather(city, 0)
            if msg.__contains__("明天天气"):
                print(msg)
                city = msg.strip().split('明')[0]
                print(city)
                return getweather(city, 1)
            return "喵喵！如果想要查询天气的话，请输入地名+今/明天天气，如【长沙明天天气】。而且猫猫学会了观星术喵咪！！可以使用【长沙赏月】、【长沙看日出】来和猫猫一起看月亮太阳哇叽！！"
        if msg.__contains__("赏月"):
            print(msg)
            city = msg.strip().split('赏')[0]
            print(city)
            return getweather(city, 3)
        if msg.__contains__("看日出"):
            print(msg)
            city = msg.strip().split('看')[0]
            print(city)
            return getweather(city, 4)


        if '重置会话' == msg.strip():
            # 清除对话内容但保留人设
            del session['msg'][1:len(session['msg'])]
            return "喵喵。。刚才发生了什么喵。。。"
        if '重置人格' == msg.strip():
            # 清空对话内容并恢复预设人设
            session['msg'] = [
                {"role": "system", "content": config_data['chatgpt']['preset']}
            ]
            return '人格已重置'
        if '查询余额' == msg.strip():
            text = ""
            for i in range(len(config_data['openai']['api_key'])):
                text = text + "Key_" + str(i + 1) + " 余额: " + str(round(get_credit_summary_by_index(i), 2)) + "美元\n"
            return text
        if '指令说明' == msg.strip():
            return "指令如下(群内需@机器人)：\n1.[重置会话] 请发送 重置会话\n2.[设置人格] 请发送 设置人格+人格描述\n3.[重置人格] 请发送 重置人格\n4.[指令说明] 请发送 " \
                   "指令说明\n注意：\n重置会话不会清空人格,重置人格会重置会话!\n设置人格后人格将一直存在，除非重置人格或重启逻辑端!"
        if msg.strip().startswith('设置人格'):
            # 清空对话并设置人设
            session['msg'] = [
                {"role": "system", "content": msg.strip().replace('设置人格', '')}
            ]
            return '人格设置成功'
        # 设置本次对话内容
        session['msg'].append({"role": "user", "content": msg})
        # 与ChatGPT交互获得对话内容
        message = chat_with_gpt(session['msg'])
        # 查看是否出错
        if message.__contains__("This model's maximum context length is 4096 token"):
            # 出错就清理一条
            del session['msg'][1:2]
            # 去掉最后一条
            del session['msg'][len(session['msg']) - 1:len(session['msg'])]
            # 重新交互
            message = chat(msg, sessionid)
        # 记录上下文
        session['msg'].append({"role": "assistant", "content": message})
        print("会话ID: " + str(sessionid))
        print("ChatGPT返回内容: ")
        print(message)
        if testpolicy(message) == False:
            print("存在敏感内容_From GPT")
            return "猫娘想到了一些不该说的，还是不说了喵。。。。QwQ"
        if message.__contains__("中国"):
            message_safe = message.replace("中国", "NULL")
            return message_safe
        return message

    except Exception as error:
        traceback.print_exc()
        return str('异常: ' + str(error))


# 获取对话session
def get_chat_session(sessionid):
    if sessionid not in sessions:
        config = deepcopy(session_config)
        config['id'] = sessionid
        sessions[sessionid] = config
    return sessions[sessionid]


def chat_with_gpt(messages):
    global current_key_index
    try:
        if not config_data['openai']['api_key']:
            return "请设置Api Key"
        else:
            if current_key_index >= len(config_data['openai']['api_key']):
                current_key_index = 0
                return "全部Key均已达到速率限制,请等待一分钟后再尝试"
            openai.api_key = config_data['openai']['api_key'][current_key_index]
        resp = openai.ChatCompletion.create(
            model=config_data['chatgpt']['model'],
            messages=messages
        )
        resp = resp['choices'][0]['message']['content']
    except openai.OpenAIError as e:
        if str(e).__contains__("Rate limit reached for default-gpt-3.5-turbo") and current_key_index < len(
                config_data['openai']['api_key']) - 1:
            # 切换key
            current_key_index = current_key_index + 1
            print("速率限制，尝试切换key")
            return chat_with_gpt(messages)
        else:
            print('openai 接口报错: ' + str(e))
            resp = str(e)
    return resp


# 生成图片
def genImg(message):
    img = text_to_image(message)
    filename = str(uuid.uuid1()) + ".png"
    filepath = config_data['qq_bot']['image_path'] + str(os.path.sep) + filename
    img.save(filepath)
    print("图片生成完毕: " + filepath)
    return filename


# 发送私聊消息方法 uid为qq号，message为消息内容
def send_private_message(uid, message):
    try:
        if testpolicy == False:
            return "猫娘想到了一些不该说的，还是不说了喵。。。。QwQ"
        if len(message) >= config_data['qq_bot']['max_length']:  # 如果消息长度超过限制，转成图片发送
            pic_path = genImg(message)
            message = "[CQ:image,file=" + pic_path + "]"
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_private_msg",
                            params={'user_id': int(uid), 'message': message}).json()
        if res["status"] == "ok":
            print("私聊消息发送成功")
        else:
            print(res)
            print("私聊消息发送失败，错误信息：" + str(res['wording']))

    except Exception as error:
        print("私聊消息发送失败")
        print(error)


# 发送私聊消息方法 uid为qq号，pic_path为图片地址
def send_private_message_image(uid, pic_path, msg):
    try:
        message = "[CQ:image,file=" + pic_path + "]"
        if msg != "":
            message = msg + '\n' + message
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_private_msg",
                            params={'user_id': int(uid), 'message': message}).json()
        if res["status"] == "ok":
            print("私聊消息发送成功")
        else:
            print(res)
            print("私聊消息发送失败，错误信息：" + str(res['wording']))

    except Exception as error:
        print("私聊消息发送失败")
        print(error)


# 发送群消息方法
def send_group_message(gid, message, uid):
    try:
        if len(message) >= config_data['qq_bot']['max_length']:  # 如果消息长度超过限制，转成图片发送
            pic_path = genImg(message)
            message = "[CQ:image,file=" + pic_path + "]"
        message = str('[CQ:at,qq=%s]\n' % uid) + message  # @发言人
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                            params={'group_id': int(gid), 'message': message}).json()
        if res["status"] == "ok":
            print("群消息发送成功")
        else:
            print("群消息发送失败，错误信息：" + str(res['wording']))
    except Exception as error:
        print("群消息发送失败")
        print(error)


# 发送群消息图片方法
def send_group_message_image(gid, pic_path, uid, msg):
    try:
        message = "[CQ:image,file=" + pic_path + "]"
        if msg != "":
            message = msg + '\n' + message
        message = str('[CQ:at,qq=%s]\n' % uid) + message  # @发言人
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                            params={'group_id': int(gid), 'message': message}).json()
        if res["status"] == "ok":
            print("群消息发送成功")
        else:
            print("群消息发送失败，错误信息：" + str(res['wording']))
    except Exception as error:
        print("群消息发送失败")
        print(error)


# 处理好友请求
def set_friend_add_request(flag, approve):
    try:
        requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/set_friend_add_request",
                      params={'flag': flag, 'approve': approve})
        print("处理好友申请成功")
    except:
        print("处理好友申请失败")


# 处理邀请加群请求
def set_group_invite_request(flag, approve):
    try:
        requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/set_group_add_request",
                      params={'flag': flag, 'sub_type': 'invite', 'approve': approve})
        print("处理群申请成功")
    except:
        print("处理群申请失败")


# openai生成图片
def get_openai_image(des):
    openai.api_key = config_data['openai']['api_key'][current_key_index]
    response = openai.Image.create(
        prompt=des,
        n=1,
        size=config_data['openai']['img_size']
    )
    image_url = response['data'][0]['url']
    print('图像已生成')
    print(image_url)
    return image_url


# 查询账户余额
def get_credit_summary():
    url = "https://chat-gpt.aurorax.cloud/dashboard/billing/credit_grants"
    res = requests.get(url, headers={
        "Authorization": f"Bearer " + config_data['openai']['api_key'][current_key_index]
    }, timeout=60).json()
    return res


# 查询账户余额
def get_credit_summary_by_index(index):
    url = "https://chat-gpt.aurorax.cloud/dashboard/billing/credit_grants"
    res = requests.get(url, headers={
        "Authorization": f"Bearer " + config_data['openai']['api_key'][index]
    }, timeout=60).json()
    return res['total_available']


if __name__ == '__main__':
    server.run(port=5555, host='0.0.0.0', use_reloader=False)
