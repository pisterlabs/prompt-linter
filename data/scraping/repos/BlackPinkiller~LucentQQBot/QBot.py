import copy
import json
import os
import traceback
import uuid
from copy import deepcopy
from flask import request, Flask
import openai
import requests
from text_to_image import text_to_image
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import random
import re
import threading
import importlib
import tiktoken
import html
import adv # 导入adv.py
import banlist # 导入banlist.py
from Slack_Bot import send_message_to_channel
from Slack_Bot import sessions as slack_sessions
from Slack_Bot import switch_message_mode as slack_switch_message_mode
from Slack_Bot import get_message_mode as slack_get_message_mode

with open("config.json", "r", encoding='utf-8') as jsonfile:
    config_data = json.load(jsonfile)
    qq_no = config_data['qq_bot']['qq_no']

with open("config_individual_wakewords.json", "r", encoding='utf-8') as jsonfile:
    awaken = json.load(jsonfile)

if config_data["VITS"]["voice_enable"] == 1:
    from CustomDic import CustomDict
    from text_to_speech import gen_speech
    from txtReader import read_txt_files
    import asyncio

def config_wakewords_reload():
    with open("config_individual_wakewords.json", "r", encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

def config_reload():
    global config_data
    with open("config.json", "r", encoding='utf-8') as jsonfile:
        config_data = json.load(jsonfile)
        qq_no = config_data['qq_bot']['qq_no']

def banlist_reload():
    importlib.reload(banlist)
    global ban_person
    global ban_group
    ban_person = banlist.ban_person
    ban_group = banlist.ban_group

def config_group_data():
    with open("config_group.json", "r", encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

def config_group_relation_data():
    with open("config_group_relation.json", "r", encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

# 读取人格文件名
def data_presets_name_fc():
    # 人格目录
    folder_path = 'presets'

    # 初始化字典
    presets_name = {}

    # 遍历文件夹内所有文件
    i = 0
    for filename in os.listdir(folder_path):
        # 如果文件是txt文件
        if filename.endswith('.txt'):
            # 去除文件名后缀
            name = os.path.splitext(filename)[0]
            # 将文件名存储到字典中
            presets_name[i] = name
            i += 1
    return presets_name
data_presets_name = data_presets_name_fc()


# 读取人格文件名空内容
def data_presets_fc():
    # 人格目录
    folder_path = 'presets'

    # 初始化字典
    presets = {}

    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path):
        # 如果文件是txt文件
        if filename.endswith('.txt'):
            # 去除文件名后缀
            name = os.path.splitext(filename)[0]
            # 将文件名和空内容存储到字典中
            presets[name] = ""
    return presets
data_presets = data_presets_fc()


# 读取人格文件内容
def data_presets_r(presets_path, presets_name):
    # 人格文件
    folder_path_flie = presets_path + presets_name + ".txt"
    # 读取文件内容
    with open(folder_path_flie, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# 读取人格文件前缀内容
def data_presets_fc1():
    # 人格目录
    folder_path1 = 'presets1'

    # 初始化字典
    presets1 = {}

    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path1):
        # 如果文件是txt文件
        if filename.endswith('.txt'):
            # 去除文件名后缀
            name = os.path.splitext(filename)[0]
            # 将文件名和内容存储到字典中
            presets1[name] = ""
    return presets1
data_presets1 = data_presets_fc1()

# 读取人格文件后缀内容
def data_presets_fc2():
    # 人格目录
    folder_path2 = 'presets2'

    # 初始化字典
    presets2 = {}

    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path2):
        # 如果文件是txt文件
        if filename.endswith('.txt'):
            # 去除文件名后缀
            name = os.path.splitext(filename)[0]
            # 将文件名和内容存储到字典中
            presets2[name] = ""
    return presets2
data_presets2 = data_presets_fc2()


# 检查JSON文件中是否存在指定的ID，如果存在则更新其对应的数据，否则创建新的键值对
def update_config_group_json(id, presets, group_mode):
    # 打开JSON文件并读取数据

    with open("config_group.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    # 检查是否存在指定的ID
    if str(id) in data:
        # 更新数据
        data[str(id)]['presets'] = presets
        data[str(id)]['group_mode'] = group_mode
    else:
        # 创建新的键值对并添加数据
        data[str(id)] = {
            'presets': presets,
            'group_mode': group_mode
        }

    # 将更新后的数据写入JSON文件
    with open('config_group.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# 检查JSON文件中是否存在指定的ID，如果存在则更新其对应的数据，否则创建新的键值对
def update_config_group_relation_json(gid, uid, relation, additional):
    # 打开JSON文件并读取数据

    with open("config_group_relation.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    # 检查是否存在指定的ID
    if str(gid) in data:
        if not '默认' in data[str(gid)]:
            # 创建新的键值对并添加数据
            data[str(gid)]['默认'] = {
                'relation': "",
                'additional': ""
            }
    else:
        # 创建新的键值对并添加数据
        data[str(gid)] = {
            '默认':{
                'relation': "",
                'additional': ""
            }
        }
    # 检查是否存在指定的ID
    if str(gid) in data:
        if str(uid) in data[str(gid)]:
            # 更新数据
            data[str(gid)][str(uid)]['relation'] = relation
            data[str(gid)][str(uid)]['additional'] = additional
        else:
            # 创建新的键值对并添加数据
            data[str(gid)][str(uid)] = {
                'relation': relation,
                'additional': additional
            }
    else:
        # 创建新的键值对并添加数据
        data[str(gid)] = {
            str(uid):{
                'relation': relation,
                'additional': additional
            }
        }

    # 将更新后的数据写入JSON文件
    with open('config_group_relation.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# 管理员QQ(可以添加权限和切换群聊对话模式，可设置多个)
moderator_qq = config_data["QBot"]["moderator_qq"]

if (len(moderator_qq) == 0):
    print("WARN:未设置moderator_qq")

# 注：安全模式即用户在无权限时无法使用自定义人格，若您不需要限制，请在config.json中设置
safe_mode = config_data["QBot"]["safe_mode"] # 安全模式

# 注：安全人格即用户在无权限（权限指是否位于advanced_users组中）时只能使用的人格，若您不需要限制，请在config.json中设置
safe_presets = config_data["QBot"]["safe_presets"] # 安全人格列表

# 若safe_presets为空
if (len(safe_presets) <= 0) and (safe_mode == 1):
    safe_mode = 0 # 强制禁用safe_mode
    print("ERROR:safe_presets列表为空，safe_mode已强制禁用")

#默认人格
data_default_preset = config_data["QBot"]["default_preset"]

# 人格简述（序号X要与config.json中的"presetX"对应）
# 例如：config.json中"preset0"对应的序号就为0
# 注意：此处的简述为人格列表展示和人格切换用！此处不应填写详细人设，详细人设应填写于config.json
# character_description = 

# 是否切换人格或重置会话时都返回Claude的消息
do_return = config_data["QBot"]["do_return"]

# 将config.json中的general_prefix（通用前置）传入
general_prefix = config_data["QBot"]["general_prefix"]

# 如果stream值不存在则赋值为False
if not config_data['qq_bot'].get('stream'):
    config_data['qq_bot']['stream'] = False
stream_enable = config_data['qq_bot']['stream']
# 全局变量, 在使用流式传输发送分段文本后添加进去的值
# 值为[sessionid][消息id].value = 发送时间戳
# 时间在默认超时90秒或在接受并发送整条gpt回复后遍历撤回
msgIDlist = {}

#语音回复
config_data_send_voice = False
    
if safe_mode == 1: # 若安全模式
    if data_default_preset in data_presets:
        session_config = {
            'msg': [
                {"role": "system", "content": data_presets_r('presets\\', safe_presets[0])} # 对话初始人格为安全人格
            ],
            "character": -2,
            "claude": config_data["qq_bot"]["claude"],
            "prefix": general_prefix
        }
    else:
        session_config = {
            'msg': [
                {"role": "system", "content": "You are a helpful assistant."} # 对话初始人格为GPT默认
            ],
            "character": -3,
            "claude": config_data["qq_bot"]["claude"],
            "prefix": general_prefix
        }
else:
    if data_default_preset in data_presets:
        session_config = {
            'msg': [
                {"role": "system", "content": data_presets_r('presets\\', data_default_preset)} # 对话初始人格为默认人格
            ],
            "character": -2,
            "claude": config_data["qq_bot"]["claude"],
            "prefix": general_prefix
        }
    else:
        session_config = {
            'msg': [
                {"role": "system", "content": ""} # 对话初始人格为GPT默认
            ],
            "character": -3,
            "claude": config_data["qq_bot"]["claude"],
            "prefix": general_prefix
        }

advanced_users = adv.advanced_users # 导入adv.py中的列表数据

# 导入banlist中的数据
ban_person = banlist.ban_person
ban_group = banlist.ban_group

sessions = {}
current_key_index = 0

# openai.api_base = "https://api.openai.com/v1"
# openai.api_base = "https://chat-gpt.aurorax.cloud/v1"
openai.api_base = config_data['openai']['endpoint']

# 创建一个服务，把当前这个python文件当做一个服务
server = Flask(__name__)


# 测试接口，可以测试本代码是否正常启动
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
    global config_data
    global config_data_presets
    global config_data_send_voice
    global data_presets_name
    global data_presets
    global data_presets1
    global data_presets2
    global config_group_data
    global awaken
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
        elif message == "重新加载配置文件":
            if (str(uid) in moderator_qq): # 若拥有权限
                # 重新加载config.json、config_individual_wakewords.json、banlist.py和config.json
                with open("config.json", "r", encoding='utf-8') as jsonfile:
                    config_data = json.load(jsonfile)
                awaken = config_wakewords_reload()
                banlist_reload()
                config_reload()
                # 重新加载预设文本
                data_presets_name = data_presets_name_fc()
                data_presets = data_presets_fc()
                data_presets1 = data_presets_fc1()
                data_presets2 = data_presets_fc2()
                print("配置文件已重新加载")
                send_private_message(uid, "配置文件已重新加载", config_data_send_voice)  # 将重载完成发送给用户
            else:
                return "错误：没有足够权限来执行此操作."
        else:
            msg_text = chat(message, 'P' + str(uid), 0, uid)  # 将消息转发给ChatGPT处理
            send_private_message(uid, msg_text, config_data_send_voice) # 将消息返回的内容发送给用户
            # 如果有需要撤回的历史消息, 则遍历历史消息字典, 并以key值(消息ID)撤回, value为时间戳
            # 如果需要key和value则需要加上.items(), 不加默认只返回key值
            if msgIDlist.get('P' + str(uid)):
                for msgID in msgIDlist.get('P' + str(uid)):
                    recall_message(msgID)
                msgIDlist.get('P' + str(uid)).clear()

    if request.get_json().get('message_type') == 'group':  # 如果是群消息
        gid = request.get_json().get('group_id')  # 群号
        uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
        nickname = request.get_json().get('sender').get('nickname')  # 发言者的昵称
        card = request.get_json().get('sender').get('card')  # 发言者的群名片
        message_id = request.get_json().get('message_id')  # 消息ID
        if card != '':
           uname = card
        else:
            uname = nickname
        message = request.get_json().get('raw_message')  # 获取原始信息
        m_gid = 'G' + str(gid)   # G+群号
        m_uid = 'G' + str(uid)  # G+发言者的qq号
        # 判断对话模式
        if not str(gid) in config_group_data():
            update_config_group_json(str(gid), "", 0)
        if config_group_data()[str(gid)]["group_mode"] != 0:
            session = get_chat_session(m_gid) # 获得对话session
        else:
            session = get_chat_session(m_uid) # 获得对话session
        data_presets_name_temp = data_presets_name  # 创建一个临时对话
        data_presets_name_temp[-1]= '自定义'
        if data_default_preset in data_presets:
            data_presets_name_temp[-2]= data_default_preset
        else:
            data_presets_name_temp[-2]= ''
        data_presets_name_temp[-3]= ''
        

        global do_awake
        # 判断当被@或触发关键词时才回答
        if data_presets_name[session['character']] in awaken.keys():  # 若当前人格有独立唤醒词
            if (str("[CQ:at,qq=%s]" % qq_no) in message) or check_strings_exist(config_data['QBot']['general_wakewords'], message) or check_strings_exist(awaken[data_presets_name_temp[int(session['character'])]], message):
                do_awake = True # 若触发，执行唤醒
            else:
                do_awake = False # 不执行唤醒
        else:  # 若当前人格没有独立唤醒词
            # 不判断是否触发独立唤醒词
            if (str("[CQ:at,qq=%s]" % qq_no) in message) or check_strings_exist(config_data['QBot']['general_wakewords'], message):
                do_awake = True # 若触发，执行唤醒
            else:
                do_awake = False # 不执行唤醒
        if do_awake == True:
            if (str(uid) in ban_person) or (str(gid) in ban_group):  # 若发言者qq号在ban_person中或群号在ban_group中
                pass  # 不处理该消息
            else:
                sender = request.get_json().get('sender')  # 消息发送者的资料
                print("收到群聊消息：")
                print(message)
                message = str(message).replace(str("[CQ:at,qq=%s] " % qq_no), '')
                if (str("戳一戳[CQ:at") in message): # 判断是否为戳一戳请求
                    poke_request = True
                else:
                    poke_request = False
                if message.strip().startswith('生成图像'):
                    message = str(message).replace('生成图像', '')
                    msg_text = chat(message, 'G' + str(gid))  # 将消息转发给ChatGPT处理
                    # 将ChatGPT的描述转换为图画
                    print('开始生成图像')
                    pic_path = get_openai_image(msg_text)
                    send_group_message_image(gid, pic_path, uid, msg_text, message_id)
                elif message.strip().startswith('直接生成图像'):
                    message = str(message).replace('直接生成图像', '')
                    print('开始直接生成图像')
                    pic_path = get_openai_image(message)
                    send_group_message_image(gid, pic_path, uid, '', message_id)
                elif message == "重新加载配置文件":
                    if (str(uid) in moderator_qq): # 若拥有权限
                        # 重新加载config.json、config_individual_wakewords.json、banlist.py和config.json
                        with open("config.json", "r", encoding='utf-8') as jsonfile:
                            config_data = json.load(jsonfile)
                        awaken = config_wakewords_reload()
                        banlist_reload()
                        config_reload()
                        # 重新加载预设文本
                        data_presets_name = data_presets_name_fc()
                        data_presets = data_presets_fc()
                        data_presets1 = data_presets_fc1()
                        data_presets2 = data_presets_fc2()
                        print("配置文件已重新加载")
                        send_group_message(gid, "配置文件已重新加载", uid, config_data_send_voice, message_id)  # 将消息转发到群用户
                    else:
                        return "错误：没有足够权限来执行此操作."
                else:
                    # 戳一戳
                    if poke_request:
                         pattern = r'\[CQ:at,qq=(\d+)\]'
                         match = re.search(pattern, message)
                         message = match.group(1)
                         msg_text = str("[CQ:poke,qq=%s]" % message)
                         at = False
                    else:
                        global general_chat
                        # 下面你可以执行更多逻辑，这里只演示与ChatGPT对话
                        # 判断对话模式
                        if not str(gid) in config_group_data():
                            update_config_group_json(str(gid), "", 0)
                        if config_group_data()[str(gid)]["group_mode"] != 0:
                            msg_text = chat(message, 'G' + str(gid), gid, uid)  # 将消息转发给ChatGPT处理（群聊共享对话）
                            sessionid = 'G' + str(gid)
                        else:
                            msg_text = chat(message, 'G' + str(uid), gid, uid)  # 将消息转发给ChatGPT处理（个人独立对话）
                            sessionid = 'G' + str(uid)
                        at = True
                    send_group_message(gid, msg_text, uid, config_data_send_voice, message_id, at)  # 将消息转发到群里
                    # 如果有需要撤回的历史消息, 则遍历历史消息字典, 并以key值(消息ID)撤回, value为时间戳
                    # 如果需要key和value则需要加上.items(), 不加默认只返回key值
                    if msgIDlist.get(sessionid):
                        for msgID in msgIDlist.get(sessionid):
                            recall_message(msgID)
                        msgIDlist.get(sessionid).clear()

    if request.get_json().get('post_type') == 'notice':  # 收到请求消息
        sub_type = request.get_json().get('sub_type')
        target_id = request.get_json().get('target_id')
        uid = request.get_json().get('sender_id')
        try:
            gid = request.get_json().get('group_id')
            message_id = request.get_json().get('message_id')
        except Exception:
            gid = None
            message_id = None
        if (sub_type == "poke") and (target_id == int(qq_no)):
            msg_text = str("[CQ:poke,qq=%s]" % uid)
            if gid == None:
                send_private_message(uid, msg_text, config_data_send_voice)
            else:
                send_group_message(gid, msg_text, uid, config_data_send_voice, message_id, False)

            send_group_message(gid, msg_text, uid, config_data_send_voice, message_id)  # 将消息转发到群里

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




# 测试接口，可以用来测试与ChatGPT的交互是否正常，用来排查问题
@server.route('/chat', methods=['post'])
def chatapi():
    requestJson = request.get_data()
    if requestJson is None or requestJson == "" or requestJson == {}:
        resu = {'code': 1, 'msg': '请求内容不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(requestJson)
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
    requestJson = request.get_data()
    if requestJson is None or requestJson == "" or requestJson == {}:
        resu = {'code': 1, 'msg': '请求内容不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(requestJson)
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
def chat(msg, sessionid, *args):
    global config_data
    global config_data_presets
    global config_data_send_voice
    global data_presets_name
    global data_presets
    global data_presets1
    global data_presets2
    global config_group_data
    global awaken
    global general_prefix
    try:
        global advanced_users
        global safe_mode
        if msg.strip() == '':
            return '您好，我是人工智能助手，如果您有任何问题，请随时告诉我，我将尽力回答。\n如果您需要重置我们的会话，请回复`重置会话`'
        # 获得对话session
        session = get_chat_session(sessionid)
        if config_data['qq_bot']['claude'] == True:
            if sessionid not in slack_sessions:
                if session['character'] == -2:
                    send_message_to_channel(message_text=data_presets_r('presets\\', data_default_preset),session_id=sessionid)
                elif session['character'] >= 0:
                    send_message_to_channel(message_text=data_presets_r('presets\\', data_presets_name[session['character']]),session_id=sessionid)
        if '语音回复' == msg.strip():
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            if config_data["VITS"]["voice_enable"] == 1:  # 管理员权限
                if str(uid) in moderator_qq:  # 管理员权限
                    if config_data_send_voice == False:
                        config_data_send_voice = True
                        print('语音回复已开启')
                        return '语音回复已开启'
                    else:
                        config_data_send_voice = False
                        print('语音回复已关闭')
                        return '语音回复已关闭'
                else:
                    return "错误：没有足够权限来执行此操作."
            else:
                return "错误：配置文件未启用语音功能."
        if '切换claude' == msg.strip().lower():
            session['claude'] = not session['claude']
            if session['claude']:
                if sessionid not in slack_sessions:
                    if session['character'] == -2:
                        send_message_to_channel(message_text=data_presets_r('presets\\', data_default_preset),session_id=sessionid)
                    elif session['character'] >= 0:
                        send_message_to_channel(message_text=data_presets_r('presets\\', data_presets_name[session['character']]),session_id=sessionid)
            return f"Claude状态： {'启用' if session['claude'] else '关闭'}\n当前模式： { '完整对话' if slack_get_message_mode() else '快速对话'}"
        if '切换claude模式' == msg.strip().lower():
            if not session['claude']:
                return "请先打开Claude模式,指令“切换claude”"
            else:
                return f"切换成功！注意：切换成功前发送的信息有可能会出错。\n当前模式： { '完整对话' if slack_switch_message_mode() else '快速对话'}"
        if 'claude模式' == msg.strip().lower():
            if not session['claude']:
                return "请先打开Claude模式,指令“切换claude”"
            else:
                return f"当前模式： { '完整对话' if slack_get_message_mode() else '快速对话'}"
        if '重置会话' == msg.strip():
            # 清除对话内容但保留人设
            if not session['claude']:
                del session['msg'][1:len(session['msg'])]
            if session['character'] == -2:
                if session['claude']:
                    slack_sessions.pop(sessionid, None)
                    send_message_to_channel(message_text=data_presets_r('presets\\', data_default_preset),session_id=sessionid)
                else:
                    session['msg'][0] = {"role": "system", "content": data_presets_r('presets\\', data_default_preset)}
            elif session['character'] >= 0:
                if session['claude']:
                        slack_sessions.pop(sessionid, None)
                        messa = send_message_to_channel(message_text=data_presets_r('presets\\', data_presets_name[session['character']]),session_id=sessionid)
                else:
                    session['msg'][0] = {"role": "system", "content": data_presets_r('presets\\', data_presets_name[session['character']])}
            if session['claude'] and do_return:
                return "会话已重置" + '\n返回内容：' + str(messa)
            else:
                return f"会话已重置"
        if '重置人格' == msg.strip():
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            if ((str(uid) in advanced_users) or (str(uid) in moderator_qq) or (safe_mode == 0)):  # 若用户在advanced_users组中或安全模式关
                if data_default_preset in data_presets:
                    if session['claude']:
                        slack_sessions.pop(sessionid, None)
                        messa = send_message_to_channel(message_text=data_presets_r('presets\\', data_default_preset),session_id=sessionid)
                    else:
                        # 清空对话内容并恢复预设人设
                        session['msg'] = [
                            {"role": "system", "content": data_presets_r('presets\\', data_default_preset)}
                        ]
                    session['character'] = -2
                    if session['claude'] and do_return:
                        return '人格已重置 当前人格：' + data_default_preset + '\n返回内容：' + str(messa)
                    else:
                        return '人格已重置 当前人格：' + data_default_preset
                else:
                    if session['claude']:
                        slack_sessions.pop(sessionid, None)
                        return '人格已重置 当前人格：' + " 无预设Claude"
                    # 清空对话内容并恢复预设人设
                    session['msg'] = [
                        {"role": "system", "content": ""}
                    ]
                    session['character'] = -3
                    return '人格已重置 当前人格：' + "ChatGPT"
            else:
                if data_default_preset in data_presets:
                    # 清空对话内容并恢复预设人设
                    if session['claude']:
                        slack_sessions.pop(sessionid, None)
                        messa = send_message_to_channel(message_text=data_presets_r('presets\\', safe_presets[0]),session_id=sessionid)
                    else:
                        session['msg'] = [
                            {"role": "system", "content": data_presets_r('presets\\', safe_presets[0])}
                        ]
                    session['character'] = -2
                    if session['claude'] and do_return:
                        return '人格已重置 当前人格：' + safe_presets[0] + '\n返回内容：' + str(messa)
                    else:
                        return '人格已重置 当前人格：' + safe_presets[0]
                else:
                    # 清空对话内容并恢复预设人设
                    if session['claude']:
                        slack_sessions.pop(sessionid, None)
                        return '人格已重置 当前人格：' + " 无预设Claude"
                    session['msg'] = [
                        {"role": "system", "content": ""}
                    ]
                    session['character'] = -3
                    return '人格已重置 当前人格：' + "ChatGPT"
        if '查询余额' == msg.strip():
            if session['claude']:
                return "Claude模式下无法查询余额"
            text = ""
            for i in range(len(config_data['openai']['api_key'])):
                text = text + "Key_" + str(i + 1) + " 用量: " + get_credit_summary_by_index(i) + "\n"
            return text
        if msg.strip().startswith('添加前置'):
            ques = msg.strip().replace('添加前置', '')
            session['prefix'] = str(ques)
            return '对话前置设置成功'
        if '移除前置' == msg.strip():
            if session['prefix'] != "":
                session['prefix'] = ""
                return "成功移除了前置."
            else:
                return "错误：没有可供移除的前置."
        if '指令说明' == msg.strip():
            return '指令如下(群内需@机器人或开头加上该人格的名字[仅预设])：\n1.[重置会话] 请发送 重置会话\n2.[切换人格] 请发送 "切换人格 人格排序号（或人格名）"(发送"人格列表"可以查看所有人格预设)\n3.[自定义人格] 请发送 "自定义人格 <该人格的描述>"\n4.[人格列表] 请发送 人格列表\n5.[重置人格] 请发送 重置人格\n6.[忘记上一条对话]请发送 忘记上一条对话\n7.[当前人格] 请发送 当前人格\n8.[指令说明] 请发送 ' \
                   '指令说明\n9.[查看群聊对话模式] 请发送 查看群聊对话模式\n10.[切换群聊对话模式] 请管理员发送 切换群聊对话模式\n11.[添加权限] 请管理员发送 "添加权限 QQ号"\n12.[切换安全模式] 请管理员发送 切换安全模式\n13.[重新加载配置文件] 热更新配置及新增（或删除）的人格文件\n14.[新增人格关系] 请发送"添加人格关系 QQ号-关系-关系前缀"\n15.[删除人格关系] 请发送"删除人格关系 QQ"\n16.[查看人格关系] 请发送"查看人格关系\n17.[切换claude] 开启/关闭Claude模式\n18.[切换claude模式] 在Claude的完整对话和快速对话模式中切换\n19.[claude模式] 查看Claude当前的对话模式"\n' \
                   '20.[添加前置] 请发送 "添加前置 前置"（可以一键在发送的对话中加上特定前置，让你无需每次对话都手动加）\n21.[移除前置] 请发送 "移除前置"\n注意：\n重置会话不会清空人格,重置人格会重置会话!\n设置人格后人格将一直存在，除非重置人格或重启逻辑端!'
        '''
        if msg.strip().startswith('/img'):
            msg = str(msg).replace('/img', '')
            print('开始直接生成图像')
            pic_path = get_openai_image(msg)
            return "![](" + pic_path + ")"
        '''
        if '忘记上一条对话' == msg.strip():
            if session['claude']:
                return "Claude不支持这个功能"
            if session['msg'][-1]["role"] != "assistant":
                if session['msg'][-1]["role"] == "system":
                    return "错误：没有可供清除的对话。"
                else:
                    return "您需要等待我回答完毕之后才能使用这个功能￣ω￣="
            else:
                del session['msg'][-2:len(session['msg'])]
                return "唔~我已经忘记了与您的上一条对话啦o(´^｀)o"
        if msg.strip().startswith('切换人格'):
            passing = False
            gid = request.get_json().get('group_id')  # 群号
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            print(request.get_json())
            ques = msg.strip().replace('切换人格', '')     # 将msg（用户输入）中的"切换人格"删除
            ques = ques.strip()       # 将msg中的空白删除
            ques_l = ques.lower()     # 将msg中的字母全部转换为小写以供识别
            if safe_mode:
                keys = []
                for x in range(0, len(safe_presets)):
                    not_found = False
                    try:
                        keyi = next(k for k, v in data_presets_name.items() if v == safe_presets[x])
                    except StopIteration:
                        # 没有匹配的键，进行相应的处理
                        not_found = True
                    if not_found == False:
                        keys.append(str(keyi))
            if ((str(uid) in advanced_users) or (str(uid) in moderator_qq) or (safe_mode == 0) or (ques_l in safe_presets) or (ques_l in keys)):  # 若用户在advanced_users组中或安全模式关
                matched = False
                for i in range(0, len(data_presets_name)-3):
                    if (ques_l == str(i) or ques_l == data_presets_name[i].lower() or ques_l == str(i) + "、" + data_presets_name[i].lower()) and ques_l != "":
                        modified = True
                        try:
                            trans_text = eval(data_presets_name[i])
                        except Exception:
                            modified = False
                        if modified:
                            if session['claude']:
                                slack_sessions.pop(sessionid, None)
                                messa = send_message_to_channel(message_text=trans_text,session_id=sessionid)
                            else:
                                session['msg'] = trans_text
                        else:
                            if session['claude']:
                                slack_sessions.pop(sessionid, None)
                                messa = send_message_to_channel(message_text=data_presets_r('presets\\', data_presets_name[i]),session_id=sessionid)
                            else:
                                session['msg'] = [
                                    {"role": "system", "content": data_presets_r('presets\\', data_presets_name[i])}
                                ]
                        session['character'] = i
                        matched = True
                        if request.get_json().get('message_type') == 'group':
                            if str(gid) in config_group_data().keys():
                                if config_group_data()[str(gid)]["group_mode"] != 0:
                                    update_config_group_json(str(gid), data_presets_name[i], config_group_data()[str(gid)]["group_mode"])
                            else:
                                update_config_group_json(str(gid), data_presets_name[i], 0)
                        if session['claude'] and do_return:
                            return '人格切换成功 当前人格：' + data_presets_name[i] + '\n返回内容：' + str(messa)
                        else:
                            return '人格切换成功 当前人格：' + data_presets_name[i]
                        break
                    elif ques_l == "": # 当输入空消息时
                        if data_default_preset in data_presets:
                            if session['claude']:
                                slack_sessions.pop(sessionid, None)
                                messa = send_message_to_channel(message_text=data_presets_r('presets\\', data_default_preset),session_id=sessionid)
                            else:
                                session['msg'] = [
                                    {"role": "system", "content": data_presets_r('presets\\', data_default_preset)} # 切换至默认人格
                                ]
                            session['character'] = -2
                            matched = True
                            if request.get_json().get('message_type') == 'group':
                                if str(gid) in config_group_data().keys():
                                    if config_group_data()[str(gid)]["group_mode"] != 0:
                                        update_config_group_json(str(gid), data_default_preset, config_group_data()[str(gid)]["group_mode"])
                                else:
                                    update_config_group_json(str(gid), data_default_preset, 0)
                            if session['claude'] and do_return:
                                return '人格切换成功 当前人格：' + data_default_preset + '\n返回内容：' + str(messa)
                            else:
                                return '人格切换成功 当前人格：' + data_default_preset
                            break
                        else:
                            if session['claude']:
                                slack_sessions.pop(sessionid, None)
                            else:
                                session['msg'] = [
                                    {"role": "system", "content": ""} # 切换至ChatGPT人格
                                ]
                            session['character'] = -3
                            matched = True
                            if request.get_json().get('message_type') == 'group':
                                if str(gid) in config_group_data().keys():
                                    if config_group_data()[str(gid)]["group_mode"] != 0:
                                        update_config_group_json(str(gid), "-3", config_group_data()[str(gid)]["group_mode"])
                                else:
                                    update_config_group_json(str(gid), "-3", 0)
                            return '人格切换成功 当前人格：' + "ChatGPT"
                            break
                # 当用户输入与上述任何一种情况不匹配时
                if matched == False:
                    if session['claude']:
                        slack_sessions.pop(sessionid, None)
                        messa = send_message_to_channel(message_text=ques,session_id=sessionid)
                    else:
                        session['msg'] = [
                            {"role": "system", "content": ques}
                        ]                                                                       # 直接将输入自定义人设写入session
                    session['character'] = -1
                    if request.get_json().get('message_type') == 'group':
                        if str(gid) in config_group_data().keys():
                            if config_group_data()[str(gid)]["group_mode"] != 0:
                                update_config_group_json(str(gid), "-1", config_group_data()[str(gid)]["group_mode"])
                        else:
                            update_config_group_json(str(gid), "-1", 0)
                    if session['claude'] and do_return:
                        return '人格切换成功 当前人格：自定义' + '\n返回内容：' + str(messa)
                    else:
                        return '人格切换成功 当前人格：自定义'
            else:
                return "错误：您的用户组无法切换指定人格列表外的人格."
        if '当前人格' == msg.strip():
            if session['character'] == -3:
                desc = "ChatGPT"
            
            elif session['character'] == -2:
                desc = data_default_preset
            
            elif session['character'] == -1:
                desc = "自定义人格"
            else:
                desc = data_presets_name[session['character']]
            return '当前人格：' + str(desc)
        if '查看群聊对话模式' == msg.strip():
            if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
                return '错误：此指令仅能在群聊内使用.'
            else:
                gid = request.get_json().get('group_id')  # 群号
                if str(gid) in config_group_data().keys():
                    if config_group_data()[str(gid)]["group_mode"] == 0:
                        return '当前模式：个人独立对话'
                    elif config_group_data()[str(gid)]["group_mode"] == 1:
                        return '当前模式：群聊共享对话'
                    elif config_group_data()[str(gid)]["group_mode"] == 2:
                        return '当前模式：群聊独立对话'
                else:
                    "错误：未记录该群聊对话模式"
        if '切换群聊对话模式' == msg.strip():
            if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
                return '错误：此指令仅能在群聊内使用.'
            else:
                gid = request.get_json().get('group_id')  # 群号
                uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
                # 检测用户是否为moderator
                if (str(uid) in moderator_qq): # 若拥有权限
                    # 重置人格
                    if data_default_preset in data_presets:
                        # 清空对话内容并恢复预设人设
                        if session['claude']:
                            slack_sessions.pop(sessionid, None)
                            send_message_to_channel(message_text=data_presets_r('presets\\', data_default_preset),session_id=sessionid)
                        else:
                            session['msg'] = [
                                {"role": "system", "content": data_presets_r('presets\\', data_default_preset)}
                            ]
                        session['character'] = -2
                        if config_group_data()[str(gid)]["group_mode"] < 2:
                            update_config_group_json(str(gid), "-2", config_group_data()[str(gid)]["group_mode"] + 1)
                        else:
                            update_config_group_json(str(gid), "-2", 0)
                    else:
                        # 清空对话内容并恢复预设人设
                        if session['claude']:
                            slack_sessions.pop(sessionid, None)
                        else:
                            session['msg'] = [
                                {"role": "system", "content": ""}
                            ]
                        session['character'] = -3
                        if config_group_data()[str(gid)]["group_mode"] < 2:
                            update_config_group_json(str(gid), "-3", config_group_data()[str(gid)]["group_mode"] + 1)
                        else:
                            update_config_group_json(str(gid), "-3", 0)
                    if config_group_data()[str(gid)]["group_mode"] == 0:
                        return '群聊对话模式切换成功，会话及人格已重置 \n当前模式：个人独立对话'
                    elif config_group_data()[str(gid)]["group_mode"] == 1:
                        return '群聊对话模式切换成功，会话及人格已重置 \n当前模式：群聊共享对话'
                    elif config_group_data()[str(gid)]["group_mode"] == 2:
                        return '群聊对话模式切换成功，会话及人格已重置 \n当前模式：群聊独立对话'
                else:
                    return "错误：没有足够权限来执行此操作."
        if msg.strip().startswith('添加人格关系'):
            if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
                return '错误：此指令仅能在群聊内使用.'
            else:
                gid = request.get_json().get('group_id')  # 群号
                uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
                # 检测用户是否为moderator
                if (str(uid) in moderator_qq): # 若拥有权限
                    ques = msg.strip().replace('添加人格关系', '')     # 将msg（用户输入）中的"添加人格关系"删除
                    ques = ques.strip()       # 将msg中的空白删除
                    # 清空对话内容并恢复预设人设
                    qq, relation, additional = ques.split('-')
                    update_config_group_relation_json(str(gid), str(qq), relation, additional)
                    return '人格关系添加完成'
                else:
                    return "错误：没有足够权限来执行此操作."
        if msg.strip().startswith('删除人格关系'):
            if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
                return '错误：此指令仅能在群聊内使用.'
            else:
                gid = request.get_json().get('group_id')  # 群号
                uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
                # 检测用户是否为moderator
                if (str(uid) in moderator_qq): # 若拥有权限
                    ques = msg.strip().replace('删除人格关系', '')     # 将msg（用户输入）中的"切换人格"删除
                    ques = ques.strip()       # 将msg中的空白删除
                    
                    # 打开JSON文件并读取数据
                    with open("config_group_relation.json", "r", encoding='utf-8') as f:
                        data = json.load(f)

                    # 删除指定键
                    if ques in config_group_relation_data()[str(gid)]:
                        del data[str(gid)][ques]
                    else:
                        return '未找到该QQ'
                    # 将更新后的数据写入JSON文件
                    with open('config_group_relation.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                        
                    return '人格关系删除完成'
                else:
                    return "错误：没有足够权限来执行此操作."
        if '查看人格关系' == msg.strip():
            if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
                return '错误：此指令仅能在群聊内使用.'
            else:
                gid = request.get_json().get('group_id')  # 群号
                uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
                if str(gid) in config_group_relation_data():
                    # 遍历指定gid下的所有uid_data，并按照指定的格式打印它们
                    uid_data = config_group_relation_data()[str(gid)]
                    count = 1
                    out = "当前人格关系：\n"
                    for uuid, values in uid_data.items():
                        relation = values['relation']
                        additional = values['additional']
                        out = out + str(uuid) + "-" + str(relation) + "-" + str(additional) + "\n"
                        count += 1
                    return out
                else:
                    return "错误：当前群组未建立关系."
        if '人格列表' == msg.strip():
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            if ((str(uid) in advanced_users) or (str(uid) in moderator_qq) or (safe_mode == 0)):
                out = "当前可切换人格：\n"
                for i in range(0, len(data_presets)):
                    out = out + str(i) + '、' + data_presets_name[i] + "\n"
                out = out + "(PS:使用'切换人格 人格排序号（或人格名）'即可完成切换)"
                return out
            else:
                keys = []
                for x in range(0, len(safe_presets)):
                    not_found = False
                    try:
                        keyi = next(k for k, v in data_presets_name.items() if v == safe_presets[x])
                    except StopIteration:
                        # 没有匹配的键，进行相应的处理
                        not_found = True
                    if not_found == False:
                        keys.append(keyi)
                out = "当前可切换人格：\n"
                for i in keys:
                    out = out + str(i) + '、' + data_presets_name[i] + "\n"
                out = out + "(PS:使用'切换人格 人格排序号（或人格名）'即可完成切换)"
                return out
        if msg.strip().startswith('自定义人格'):
            gid = request.get_json().get('group_id')  # 群号
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            if ((str(uid) in advanced_users) or (str(uid) in moderator_qq) or (safe_mode == 0)): # 若用户在advanced_users组中或安全模式关
                # 清空对话并设置人设
                if session['claude']:
                    slack_sessions.pop(sessionid, None)
                    messa = send_message_to_channel(message_text=msg.strip().replace('自定义人格', ''),session_id=sessionid)
                else:
                    session['msg'] = [
                        {"role": "system", "content": msg.strip().replace('自定义人格', '')}
                    ]
                session['character'] = -1
                if request.get_json().get('message_type') == 'group':
                    if str(gid) in config_group_data().keys():
                        if config_group_data()[str(gid)]["group_mode"] != 0:
                            update_config_group_json(str(gid), "-1", config_group_data()[str(gid)]["group_mode"])
                    else:
                        update_config_group_json(str(gid), "-1", 0)
                if session['claude'] and do_return:
                    return '自定义人格设置成功\n（提示：下次切换或重置人格前都会保持此人设）' + '\n返回内容：' + str(messa)
                else:
                    return '自定义人格设置成功\n（提示：下次切换或重置人格前都会保持此人设）'
            else:
                return "错误：没有足够权限来执行此操作."
        if msg.strip().startswith('添加权限'):
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            ques = msg.strip().replace('添加权限', '') # 去除消息中的'添加权限'
            ques = ques.strip()  # 去除空格
            # 检测用户是否为moderator
            if (str(uid) in moderator_qq): # 若拥有权限
                if ques == '':  # 若未键入任何uid
                    return "错误：输入值不能为空白."
                elif ques in advanced_users:
                    return "错误：此用户已经在advanced_users组中."
                else:
                    try:
                        int(ques) # 检测键入值是否为数字
                    except ValueError: # 若不是
                        return "错误：输入值必须为QQ号"
                    adv.advanced_users.append(str(ques)) # 添加该值
                    with open("adv.py", "w",encoding='utf-8') as f: # 打开adv.py
                        f.write("advanced_users = " + str(adv.advanced_users)) # 写入新的列表
                    return "用户权限添加成功"
            else:
                return "错误：没有足够权限来执行此操作."
        if msg.strip().startswith('删除权限'):
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            ques = msg.strip().replace('删除权限', '') # 去除消息中的'删除权限'
            ques = ques.strip()  # 去除空格
            # 检测用户是否为moderator
            if (str(uid) in moderator_qq): # 若拥有权限
                if ques == '':
                    return "错误：输入值不能为空白."
                elif ques not in advanced_users:
                    return "错误：此用户不在advanced_users组中."
                else:
                    if ques in moderator_qq:
                        return "错误：无权移除该用户的权限组."
                    else:
                        try:
                            int(ques) # 检测键入值是否为数字
                        except ValueError: # 若不是
                            return "错误：输入值必须为QQ号"
                        adv.advanced_users.remove(str(ques)) # 去除该值
                        with open("adv.py", "w", encoding='utf-8') as f:  # 打开adv.py
                            f.write("advanced_users = " + str(adv.advanced_users)) # 写入新的列表
                        return "用户权限删除成功"
            else:
                return "错误：没有足够权限来执行此操作."
        if '切换安全模式' == msg.strip():
            uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
            # 检测用户是否为moderator
            if (str(uid) in moderator_qq): # 若拥有权限
                if safe_mode == 0:
                    safe_mode = 1
                    return '已开启人格切换限制（全局生效，不会写入）'
                else:
                    safe_mode = 0
                    return '已关闭人格限制（全局生效）'
            else:
                return "错误：没有足够权限来执行此操作."
        # 获取信息
        gid = request.get_json().get('group_id')  # 群号
        uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
        nickname = request.get_json().get('sender').get('nickname')  # 发言者的昵称
        card = request.get_json().get('sender').get('card')  # 发言者的群名片
        if card != '':
            uname = card
        else:
            uname = nickname
        # 设置时间
        if len(session['msg']) < 2:
            session['msg'].append(None)
        session['msg'][1] = {"role": "system", "content": "current time is:" + get_bj_time()}
        if session['prefix'] != "":
            message_edited = session['prefix'] + msg
        else:
            message_edited = msg
        # 设置本次对话内容
        if request.get_json().get('message_type') == 'group':
            if config_group_data()[str(gid)]["group_mode"] != 0: # 判断是否群聊
                if session['character'] < 0 and config_group_data()[str(gid)]["presets"] in data_presets:
                    session['msg'] = [
                        {"role": "system", "content": data_presets_r('presets\\', config_group_data()[str(gid)]["presets"])}
                    ]
                    session['character'] = 0
            if config_group_data()[str(gid)]["group_mode"] == 2: # 判断是否群聊独立
                if session['character'] >= 0:
                    if config_group_data()[str(gid)]["presets"] in data_presets1.keys():
                        msg_prefix = data_presets_r('presets1\\', config_group_data()[str(gid)]["presets"])
                    else:
                        msg_prefix = "现在说话的是 $user_name$ ，$user_name$ 说:“"
                    if config_group_data()[str(gid)]["presets"] in data_presets2.keys():
                        msg_suffix = data_presets_r('presets2\\', config_group_data()[str(gid)]["presets"])
                    else:
                        msg_suffix = "”。"
                    msg_prefix = msg_prefix.replace('$user_qq$', str(uid))
                    msg_prefix = msg_prefix.replace('$user_name$', str(uname))
                elif session['character'] == -2:
                    if data_default_preset in data_presets1.keys():
                        msg_prefix = data_presets_r('presets1\\', data_default_preset)
                    else:
                        msg_prefix = ""
                    if data_default_preset in data_presets2.keys():
                        msg_suffix = data_presets_r('presets2\\', data_default_preset)
                    else:
                        msg_suffix = ""
                    msg_prefix = msg_prefix.replace('$user_qq$', str(uid))
                    msg_prefix = msg_prefix.replace('$user_name$', str(uname))
                else:
                    msg_prefix = ""
                    msg_suffix = ""
                if session['character'] >= 0:
                    # 检查uid键是否存在于gid中
                    if str(gid) in config_group_relation_data() and str(uid) in config_group_relation_data()[str(gid)]:
                        # 使用关系生成消息前缀
                        msg_prefix = msg_prefix.replace('$user_rel$', config_group_relation_data()[str(gid)][str(uid)]["relation"])
                        if session['claude']:
                            message_edited = msg_prefix + msg + msg_suffix + config_group_relation_data()[str(gid)][str(uid)]["additional"]
                        else:
                            if session['prefix'] != "":
                                session['msg'].append({"role": "user", "content": session['prefix'] + msg_prefix + msg + msg_suffix + config_group_relation_data()[str(gid)][str(uid)]["additional"]})
                            else:
                                session['msg'].append({"role": "user", "content": msg_prefix + msg + msg_suffix + config_group_relation_data()[str(gid)][str(uid)]["additional"]})
                        
                    elif '默认' in config_group_relation_data().get(str(gid), {}):
                        # 使用默认关系生成消息前缀
                        msg_prefix = msg_prefix.replace('$user_rel$', config_group_relation_data()[str(gid)]['默认']["relation"])
                        if session['claude']:
                            message_edited = msg_prefix + msg + msg_suffix + config_group_relation_data()[str(gid)]['默认']["additional"]
                        else:
                            if session['prefix'] != "":
                                session['msg'].append({"role": "user", "content": session['prefix'] + msg_prefix + msg + msg_suffix + config_group_relation_data()[str(gid)]['默认']["additional"]})
                            else:
                                session['msg'].append({"role": "user", "content": msg_prefix + msg + msg_suffix + config_group_relation_data()[str(gid)]['默认']["additional"]})
                    else:
                        # 没有找到关系信息，不使用关系生成消息前缀
                        msg_prefix = msg_prefix.replace('$user_rel$', "")
                        if session['claude']:
                            message_edited = msg_prefix + msg + msg_suffix
                        else:
                            if session['prefix'] != "":
                                session['msg'].append({"role": "user", "content": session['prefix'] + msg_prefix + msg + msg_suffix})
                            else:
                                session['msg'].append({"role": "user", "content": msg_prefix + msg + msg_suffix})
                else:
                    msg_prefix = msg_prefix.replace('$user_name$', str(uname)).replace('$user_rel$', "")
                    if session['claude']:
                        message_edited = msg_prefix + msg + msg_suffix
                    else:
                        if session['prefix'] != "":
                            session['msg'].append(
                                {"role": "user", "content": session['prefix'] + msg_prefix + msg + msg_suffix})
                        else:
                            session['msg'].append({"role": "user", "content": msg_prefix + msg + msg_suffix})
            else:
                if not session['claude']:
                    session['msg'].append({"role": "user", "content": msg})

        else:
            if not session['claude']:
                session['msg'].append({"role": "user", "content": msg})
        # 检查是否超过tokens限制
        if not session['claude']:
            while num_tokens_from_messages(session['msg']) > config_data['chatgpt']['max_tokens']:
                # 当超过记忆保存最大量时，清理一条
                del session['msg'][2:3]
            print("上下文：")
            print(session['msg'])
        # 与ChatGPT交互获得对话内容
        if session['claude']:
            message = send_message_to_channel(message_text=message_edited,session_id=sessionid)
        else:
            message = chat_with_gpt(session['msg'], *args, sessionid)
        # 记录上下文
        if not session['claude']:
            session['msg'].append({"role": "assistant", "content": message.removesuffix(config_data['qq_bot'].get('page_finish_symbol'))})
        print("会话ID: " + str(sessionid))
        print("ChatGPT返回内容: ")
        print(html.unescape(message))
        return html.unescape(message)
    except Exception as error:
        traceback.print_exc()
        return str('异常: ' + str(error))


# 获取北京时间
def get_bj_time():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    # 北京时间
    beijing_now = utc_now.astimezone(SHA_TZ)
    fmt = '%Y-%m-%d %H:%M:%S'
    now_fmt = beijing_now.strftime(fmt)
    return now_fmt


# 获取对话session
def get_chat_session(sessionid):
    if sessionid not in sessions:
        config = deepcopy(session_config)
        config['id'] = sessionid
        sessions[sessionid] = config
    return sessions[sessionid]


def chat_with_gpt(messages, *args):
    global current_key_index
    max_length = len(config_data['openai']['api_key']) - 1
    try:
        if not config_data['openai']['api_key']:
            return "请设置Api Key"
        else:
            if current_key_index > max_length:
                current_key_index = 0
                return "全部Key均已达到速率限制,请等待一分钟后再尝试"
            openai.api_key = config_data['openai']['api_key'][current_key_index]
            # 如果stream开启则开始使用stream方法
            if config_data['qq_bot'].get('stream') if config_data['qq_bot'].get('stream') else stream_enable:
                print("开始stream模式")
                bot_info = get_bot_info()
                nick_name = bot_info['nickname']
                bot_qq = bot_info['user_id']
                import time
                stream_resp = chat_completion(stream=True, messages=messages)
                # 定义收集chunk的变量
                # full_reply_content 用于返回全部文本
                # text_chunk 把回复分块后的chunk, 默认为->超过75字符后遇到 \n 换行符号或句号
                # chunk_collection 用于比对数据, 捕捉代码块
                full_reply_content = ""
                text_chunk = ""
                chunk_collection = []
                # 提前判断本session发送的需要撤回的text_chunk列表是否存在
                # 不赋值的话会报错, >>已经再下方使用 if 避免了报错, 但是需要赋空值才能放入数据
                if not msgIDlist.get(args[2]):
                    msgIDlist[args[2]] = {}
                qq_response = None

                typing_status = config_data['qq_bot'].get('typing_status') if config_data['qq_bot'].get('typing_status') else " ✏️ 正在输入... ◕‿◕"
                # 页码符号只会在私聊中发送, 并且可关闭, 默认为开启
                page_suffix_enable = config_data['qq_bot'].get('page_suffix') if config_data['qq_bot'].get(
                    'page_suffix') else False
                # 页码符号
                page_stream_symbol = config_data['qq_bot'].get('page_stream_symbol') if config_data['qq_bot'].get('page_stream_symbol') else "..."  # 💬
                page_finish_symbol = config_data['qq_bot'].get('page_finish_symbol') if config_data['qq_bot'].get('page_finish_symbol') else ""  # "🔚"
                # 等待时间, 默认为20秒
                # 每次触发返回的chunk字符量, 默认为75个字符
                # 任一条件满足就会返回
                time_cap = config_data['qq_bot'].get('time_cap') if config_data['qq_bot'].get('time_cap') else 20
                chunk_chars_limit = config_data['qq_bot'].get('chunk_chars') if config_data['qq_bot'].get('chunk_chars') else 150
                # 触发符号, 默认为换行符号等
                trigger_symbol = config_data['qq_bot'].get('trigger_symbol') if config_data['qq_bot'].get('trigger_symbol') else ['\n\n', '\n', '.', '。']

                code_symbol = "```"
                # 开始流式传输...
                # chunk为每次gpt返回的一个单个类似token的东西
                # chunk不一定为1个字符
                start_time = time.time()
                for chunk in stream_resp:
                    # 复用当前chunk数据, 所有提前赋一下值让代码好读
                    chunk_content = chunk['choices'][0]['delta'].get('content', '')
                    # 使用本次chunk更新收集chunk的变量
                    chunk_collection.append(chunk)
                    text_chunk += ''.join(chunk_content)
                    full_reply_content += ''.join(chunk_content)
                    # 根据配置文件中的 "qq_bot" -> "chunk_chars" 参数，
                    # 将响应分成较小的块。这个参数指定用于将响应拆分成块的字符。
                    # 默认为75, 可更改.
                    code_paired = True if full_reply_content.count(code_symbol) % 2 == 0 else False
                    form_found = True if '|' in full_reply_content and '|\n\n' not in full_reply_content else False
                    if (len(text_chunk) >= chunk_chars_limit or (time.time() - start_time >= time_cap)) and text_chunk[-1] in trigger_symbol and code_paired and not form_found:
                        finish_reason = chunk["choices"][0]["finish_reason"]
                        if finish_reason == "stop":
                            if page_suffix_enable and args[0] == 0 and len(full_reply_content) > chunk_chars_limit * 2:
                                full_reply_content += f"{page_finish_symbol}"
                            resp = full_reply_content
                            set_group_card(args[0], qq_no, nick_name)
                            print("传输完成, stream模式结束")
                            return resp
                        # 去掉行尾空格, 如果有的话,格式美观 :)
                        text_chunk = text_chunk[:-2] + text_chunk[-2:].replace('\n', '')
                        if args[0] != 0 and config_group_data()[str(args[0])]["group_mode"] >= 1:
                            # 首先判断是否为群聊模式, 在套娃式调用到本方法的时候, 会传递uid和gid, 私聊模式传递的gid为0
                            # 如果有群id并且群id所在模式为群聊共享模式, 则执行, 否则不发送消息
                            # args[0],[1],[2], 分别是 gid, uid, sessionid, 这也是不改动源代码我想到比较好的方法=
                            set_group_card(args[0], qq_no, nick_name + typing_status)
                            qq_response = send_group_message(args[0],
                                                             text_chunk,
                                                             args[1] if len(msgIDlist.get(args[2]) if msgIDlist.get(args[2]) else {}) == 0 else "",
                                                             config_data_send_voice, 0)
                            print("太久了! 发送分段消息")
                        elif args[0] == 0:
                            text_chunk += f"{page_stream_symbol}"
                            qq_response = send_private_message(args[1], text_chunk, config_data_send_voice)
                        # 如果有消息发送并且成功了, 将消息的id, 发送时间填入msgIDlist中对应的sessionid的字典中
                        if qq_response and qq_response.get('data').get('message_id'):
                            msgIDlist[args[2]].update({qq_response.get('data').get('message_id'): time.time()})
                        text_chunk = ""
                        start_time = time.time()
                        # 结束分块判断...
                    # 如果本session中有历史发送的消息, 则循环查看他们是否超过90秒, 超出则撤回
                    # 不使用get语句的话会报错, 先判断再执行
                    if msgIDlist.get(args[2]):
                        temp_msg_list = copy.deepcopy(msgIDlist.get(args[2]))
                        for msgID, time_stamp in temp_msg_list.items():
                            if time.time() - time_stamp > 90:
                                recall_message(msgID)
                                if msgID in msgIDlist.get(args[2]):
                                    del msgIDlist[args[2]][msgID]
                # 结束流式传输
                if page_suffix_enable and args[0] == 0 and len(full_reply_content) > chunk_chars_limit * 2:
                    full_reply_content += f"{page_finish_symbol}"
                resp = full_reply_content
                set_group_card(args[0], qq_no, nick_name)
                print("stream模式结束")
            else:
                resp = chat_completion(stream=False, messages=messages)
                resp = resp['choices'][0]['message']['content']
    except openai.OpenAIError as e:
        if str(e).__contains__("Rate limit reached for default-gpt-3.5-turbo") and current_key_index <= max_length:
            # 切换key
            current_key_index = current_key_index + 1
            print("速率限制，尝试切换key")
            return chat_with_gpt(messages)
        elif str(e).__contains__("Your access was terminated due to violation of our policies") and current_key_index <= max_length:
            print("请及时确认该Key: " + str(openai.api_key) + " 是否正常，若异常，请移除")
            if current_key_index + 1 > max_length:
                return str(e)
            else:
                print("访问被阻止，尝试切换Key")
                # 切换key
                current_key_index = current_key_index + 1
                return chat_with_gpt(messages)
        else:
            print('openai 接口报错: ' + str(e))
            resp = str(e)
    return resp


def recall_message(message_id) -> None:
    """
    撤回函数, 会在控制台打印出状态
    Args:
        message_id: 要撤回的消息id
    """
    print("recall message. id: {} status:{}".format(message_id, requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/delete_msg",
                                                                             params={'message_id': message_id})))

def get_bot_info():
    """
    获取Bot的昵称
    """
    data = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/get_login_info").json()
    if data['status'] == 'ok':
        return data['data']
    return {'nickname': '','user_id': qq_no}

def set_group_card(group_id, user_id, nick_name) -> None:
    """
    设置群名片
    Args:
        group_id: 群号
        user_id: 用户qq号
        nick_name: 群名片
    """
    params = {'group_id': group_id, 'user_id': user_id, 'card': nick_name}
    requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/set_group_card", params=params)


def chat_completion(stream: False, messages: ""):
    """
    为代码复用性而单独写的一个函数
    Args:
        stream:bool: 是否为流式传输
        messages:str: 用户输入文本

    Returns:
        resp:流式传输的openai Generator对象
    """
    resp = openai.ChatCompletion.create(
        model=config_data['chatgpt']['model'],
        messages=messages,
        temperature=config_data['chatgpt']['temperature'],
        top_p=config_data['chatgpt']['top_p'],
        presence_penalty=config_data['chatgpt']['presence_penalty'],
        frequency_penalty=config_data['chatgpt']['frequency_penalty'],
        stream=stream
    )
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
def send_private_message(uid, message, send_voice):
    try:
        if send_voice:  # 如果开启了语音发送
            voice_path = asyncio.run(
                gen_speech(message, config_data['qq_bot']['voice'], config_data['qq_bot']['voice_path']))
            message = "[CQ:record,file=file://" + voice_path + "]"
        if len(message) >= config_data['qq_bot']['max_length'] and not send_voice:  # 如果消息长度超过限制，转成图片发送
            pic_path = genImg(message)
            message = "[CQ:image,file=" + pic_path + "]"
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_private_msg",
                            params={'user_id': int(uid), 'message': message}).json()
        if res["status"] == "ok":
            print("私聊消息发送成功")
            return res
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
def send_group_message(gid, message, uid, send_voice, message_id, at=True):
    try:
        if len(message) >= config_data['qq_bot']['max_length']:  # 如果消息长度超过限制，转成图片发送
            pic_path = genImg(message)
            pic_message = "[CQ:image,file=" + pic_path + "]"
            res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                                params={'group_id': int(gid), 'message': pic_message}).json()
            if res["status"] == "ok":
                print("群图片发送成功")
                return res
            else:
                print("群图片发送失败，错误信息：" + str(res['wording']))
        else:
            message_message = message
            if at and uid:
                message_message = str('[CQ:at,qq=%s]\n' % uid) + message_message  # @发言人
            res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                                params={'group_id': int(gid), 'message': message_message}).json()
            if res["status"] == "ok":
                print("群消息发送成功")
                return res
            else:
                print("群消息发送失败，错误信息：" + str(res['wording']))
        if send_voice:  # 如果开启了语音发送
            voice_message = message
            voice_path = asyncio.run(
                gen_speech(voice_message, config_data['qq_bot']['voice'], config_data['qq_bot']['voice_path']))
            voice_message = "[CQ:record,file=file:///" + voice_path + "]"
            res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                                params={'group_id': int(gid), 'message': voice_message}).json()
            if res["status"] == "ok":
                print("群语音发送成功")
                return res
            else:
                print("群语音发送失败，错误信息：" + str(res['wording']))
    except Exception as error:
        print("群消息发送失败")
        print(error)


# 发送群消息图片方法
def send_group_message_image(gid, pic_path, uid, msg, message_id):
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
    return checkBalance(config_data['openai']['api_key'][current_key_index])

# 查询账户余额
def get_credit_summary_by_index(index):
    return checkBalance(config_data['openai']['api_key'][index])


def checkBalance(key):
    try:
        import datetime
        curr_time = datetime.datetime.now()
        next_month = curr_time.replace(day=28) + datetime.timedelta(days=4)

        last_day_of_month = (next_month - datetime.timedelta(days=next_month.day)).strftime("%Y-%m-%d")
        first_day_of_month = curr_time.replace(day=1).strftime("%Y-%m-%d")
        usage_url = f"{'https://api.openai.com/dashboard/billing/usage'}?start_date={first_day_of_month}&end_date={last_day_of_month}"
        next_day = curr_time + datetime.timedelta(days=1)
        usage_url_today = f"{'https://api.openai.com/dashboard/billing/usage'}?start_date={curr_time.strftime('%Y-%m-%d')}&end_date={next_day.strftime('%Y-%m-%d')}"
        subscribtion_url = "https://api.openai.com/dashboard/billing/subscription"
        try:
            usage_data = _get_billing_data(usage_url,key)
            usage_data_today = _get_billing_data(usage_url_today,key)
            subscription_data = _get_billing_data(subscribtion_url, key)
        except Exception as e:
            print(f"获取API使用情况失败:" + str(e))
            return "<font size=\"1\">**获取API使用情况失败**</font>"
        rounded_usage = "{:.2f}".format(usage_data["total_usage"] / 100)
        rounded_usage_today = "{:.2f}".format(usage_data_today["total_usage"] / 100)
        rounded_subscription = "{:.2f}".format(subscription_data["hard_limit_usd"])

        return f"\n\u3000总订阅额度: \u3000 ${rounded_subscription} 美刀\n\u3000本月使用额度: \u3000 ${rounded_usage} 美刀\n\u3000今日使用额度: \u3000 ${rounded_usage_today} 美刀"
    except Exception as e:
        print(f"获取API使用情况失败:" + str(e))
        return "获取API使用情况失败"


def _get_billing_data(billing_url, key):
    response = requests.get(
                        billing_url,
                        headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {key}",
                                },
                        timeout=30,
                        )
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )

def _get_subscription(billing_url,key):
    response = requests.get(
        billing_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        timeout=30,
    )
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )




# 计算消息使用的tokens数量
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # 如果name字段存在，role字段会被忽略
                    num_tokens += -1  # role字段是必填项，并且占用1token
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(f"""当前模型不支持tokens计算: {model}.""")

# 判断关键字数组是否存在字符串中
def check_strings_exist(str_list, target_str):
    for s in str_list:
        if s.lower() in target_str.lower():
            return True
    return False





if __name__ == '__main__':
    config_port = config_data["qq_bot"]["cqhttp_post_port"]
    server.run(port=config_port, host='0.0.0.0', use_reloader=False)
