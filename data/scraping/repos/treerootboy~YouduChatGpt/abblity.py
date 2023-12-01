'''
TODO:
[ ] 会议室预定算法优化
    - 思路：
      1. 将预定时段按每半小时拆分，得到一个时段列表，时段序号为key，时段开始时间、结束时间、是否已预定为value
      2. 通过全时段的dict.update方法更新已预定的时段
      3. 创建一个会议室对象，包含会议室名称、容纳人数、时段列表，提供预定方法、取消预定方法、获取可用时段方法
'''

import datetime
from functools import wraps
import time
import openai
import json
import requests
import pandas as pd

import entapp.client as app
from entapp.message import *
from entapp.aes import AESCrypto
from entapp.aes import generate_signature
from entapp.utils import *
from environs import Env
env = Env()
env.read_env()

YOUDU_BUIN = env.int('YOUDU_BUIN')
YOUDU_APP_ID = env.str('YOUDU_APP_ID')
YOUDU_AES_KEY = env.str('YOUDU_AES_KEY')
YOUDU_CALLBACK_TOKEN = env.str('YOUDU_CALLBACK_TOKEN')
YOUDU_ADDRESS = env.str('YOUDU_ADDRESS', '192.168.8.180:7080')
YOUDU_DOWNLOAD_DIR = env.str('YOUDU_DOWNLOAD_DIR', './download')

HEADLESS = True
WORK_ID = env.str('WORK_ID')
WORK_PASSWORD = env.str('WORK_PASSWORD')

client = app.AppClient(YOUDU_BUIN, YOUDU_APP_ID, YOUDU_AES_KEY, YOUDU_ADDRESS)

# 写一个输出方法返回值的装饰器，要显示实际调用的方法名
def print_result(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        print(f'调用方法：{func.__name__}，返回值：{result}')
        return result
    return wrapper

def current_time():
    now = datetime.datetime.utcnow()+datetime.timedelta(hours=8)
    beijing_tz = datetime.timezone(datetime.timedelta(hours=8))
    beijing_time = now.astimezone(beijing_tz)
    return beijing_time.isoformat()

def auto_select_model(messages):
    tokens = num_tokens(messages)
    print('tokens number is ',tokens)
    if tokens > 16384:
        return "gpt-4-32k"
    elif tokens > 4096:
        return "gpt-3.5-turbo-16k"
    else:
        return "gpt-3.5-turbo-0613"

import tiktoken
def num_tokens(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value if not is_instance(value, dict) else json.dumps(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



functions = []

functions.append({
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA, defaults to Shenzhen, China",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
})
@print_result
async def get_current_weather(location="shenzhen", unit="celsius", user=None):
    
    # 翻译成英文
    location = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": f"translate '{location}' to English, and just show the result only, no other words"
            },
        ]
    ).choices[0].message.content
    
    """use weatherapi.com get the current weather in a given location"""
    url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": "af11f7f07cf74d71877114731231406", "q": location, "aqi": "no"}
    print(location, unit, user)
    data = requests.get(url, params=params).json()
    print(location, data)
    weather_info = {
        "location": location,
        "temperature": data["current"]["temp_" + unit[0]],
        "forecast": ["sunny", "windy"],
    }
    return weather_info


functions.append({
    "name": "get_audit_accounts",
    "description": "Get accounts list file for audit",
    "parameters": {
        "type": "object",
        "properties": {}
    }
})
from plugins.auditscript import auditaccount
import uuid
@print_result
async def get_audit_accounts(user=None):
    if user is None or user == '':
        raise Exception('user is empty')
    
    _file = auditaccount.run().get('accounts_file')
    media_id = client.upload_file('xslx', uuid.uuid1(), _file)
    client.send_msg(Message(
        user, 
        MESSAGE_TYPE_FILE, 
        FileBody(media_id)
    ))
    return {
        'accounts_file': auditaccount.run().get('accounts_file'),
        'message': f'成功获取账号列表，已发送至您的邮箱，请注意查收，文件名是：{_file}'
    }
    
functions.append({
    "name": "get_staff_info",
    "description": "get staff info, search by name",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "staff name",
            }
        },
        "required": ["name"],
    }
})
@print_result
async def get_staff_info(name, user=None):
    if user is None or user == '':
        raise Exception('user is empty')
    
    data = requests.get(f'https://apsm.addcn.com/api/v1/users?name={name}').json()
    return data.get('data', {}).get('data', [])[0]


# 定会议室
functions.append({
    "name": "book_meetingroom",
    "description": "order meeting room",
    "parameters": {
        "type": "object",
        "properties": {
            "theme": {
                "type": "string",
                "description": "meeting theme",
            },
            "name": {
                "type": "string",
                "description": "booking room name",
            },
            "date": {
                "type": "string",
                "description": "booking date, default is today, format is YYYY-MM-DD",
            },
            "at": {
                "type": "string",
                "description": "booking time, default is the next 30-minute mark, format is HH:mm",
            },
            "duration": {
                "type": "integer",
                "description": "booking duration, step is 0.5 hour, default is 1 hour",  
            }
        },
        "required": ["at","name"],
    }
})
from playwright.async_api import async_playwright
import re
@print_result
async def book_meetingroom(name, date=time.strftime("%Y-%m-%d", time.localtime()), at=time.strftime("%H:%M", time.localtime()), duration=0.5, theme=None, user=None):
    at = next_30_minute_mark(at)
    
    data, cookies = await get_meetingroom_info(date)
    if not data:
        return {
            'message': '抱歉，获取会议室列表失败，请稍后重试'
        }
    
    _end = minutes_to_time(time_to_minutes(at) + int(duration * 60))
    macther = re.compile(r'.*(\d{4}|贵宾|休闲).*')
    _name = macther.match(name).group(1) if macther.match(name) is not None else name
    
    booked_room = [
        booking for booking in data 
        if macther.match(booking['room_name']).group(1) == _name 
        and is_time_overlap(at, _end, booking['start'], booking['end'])
    ]
    if len(booked_room) > 0:
        return {
            'message': f'抱歉，{name}在{date} {at}到{date} {_end}已经被预定了，请选择其他时间段或其他会议室'
        }
    booking_data = {
        "meeting_theme": theme or f"{user}的会议",
        "is_video": "0",
        "peopleNum": 2,
        "timeStart": [k for k, v in enumerate(get_time_slot()) if v['start']==at][0],
        "timeEnd": [k for k, v in enumerate(get_time_slot()) if v['start']==_end][0],
        "user":"",
        "area": [k for k, v in get_room().items() if macther.match(v['room_name']).group(1) == _name][0],
        "day": date
    }
    response = requests.post('https://oa.addcn.com/Home/Booked/booking', cookies=cookies, data=booking_data)
    booking_result = response.json()
    
    return {
        'booking_data': {
            'theme': theme,
            'name': name,
            'date': date,
            'at': at,
            'duration': duration,
        },
        'message': booking_result['msg']
    }

async def get_meetingroom_info(date):
    playwright = await async_playwright().start()
    context = await playwright.chromium.launch_persistent_context(headless=HEADLESS, user_data_dir='./userdata', locale='zh-TW')
    
    await oa_login(context)
    _cookies = await context.cookies()
    cookies = {}
    for cookie in _cookies:
        cookies[cookie['name']] = cookie['value']
    
    page = await context.new_page()
    await page.goto(f'https://oa.addcn.com/Home/Booked/index/selectDay/{date}.html')
    _json = await page.locator('.json-data').first.text_content()
    await context.close()
    
    data = None
    if _json:
        data = format_meetingroom_data(json.loads(_json))
    
    return (data, cookies)

# 下一个30分钟的时间点
def next_30_minute_mark(time_str):
    minutes = time_to_minutes(time_str)
    return minutes_to_time(minutes + (30 - minutes % 30) % 30)

# 检查时间段是否重叠，参数为字符串，格式为HH:mm
def is_time_overlap(time1_start, time1_end, time2_start, time2_end):
    return time_to_minutes(time1_start) < time_to_minutes(time2_end) and time_to_minutes(time1_end) > time_to_minutes(time2_start)
    
        
# 将时间字符串转换成分钟
def time_to_minutes(time_str):
    return int(time_str.split(':')[0]) * 60 + int(time_str.split(':')[1])

# 将分钟数转换成时间字符串
def minutes_to_time(minutes):
    return f"{str(minutes // 60).zfill(2)}:{str(minutes % 60).zfill(2)}"

async def oa_login(context):
    page = await context.new_page()
    await page.goto('https://oa.addcn.com/')
    if await page.locator('#work_id').count()>0:
        await page.locator('#work_id').fill(WORK_ID)
        await page.locator('#pwd').fill(WORK_PASSWORD)
        await page.locator('#rmbUser').click()
        await page.get_by_text('登 入').click()

def format_meetingroom_data(data):
    time_slot = get_time_slot()
    room = get_room()
    for booking in data:
        booking['start'] = time_slot[booking['start']]['start']
        booking['end'] = time_slot[booking['end']]['end']
        booking['room_name'] = room[booking['area']]['room_name']
        booking['room_limit'] = room[booking['area']]['room_limit']
        del booking['area']
    return data

def get_time_slot():
    return [{"start":"08:30","end":"09:00"},{"start":"09:00","end":"09:30"},{"start":"09:30","end":"10:00"},{"start":"10:00","end":"10:30"},{"start":"10:30","end":"11:00"},{"start":"11:00","end":"11:30"},{"start":"11:30","end":"12:00"},{"start":"12:00","end":"12:30"},{"start":"12:30","end":"13:00"},{"start":"13:00","end":"13:30"},{"start":"13:30","end":"14:00"},{"start":"14:00","end":"14:30"},{"start":"14:30","end":"15:00"},{"start":"15:00","end":"15:30"},{"start":"15:30","end":"16:00"},{"start":"16:00","end":"16:30"},{"start":"16:30","end":"17:00"},{"start":"17:00","end":"17:30"},{"start":"17:30","end":"18:00"},{"start":"18:00","end":"18:30"},{"start":"18:30","end":"19:00"},{"start":"19:00","end":"19:30"},{"start":"19:30","end":"20:00"},{"start":"20:00","end":"20:30"},{"start":"20:30","end":"21:00"},{"start":"21:00","end":"21:30"},{"start":"21:30","end":"22:00"}]        

def get_room():
    return {"2":{"room_name":"会议室1202","room_limit":6},"3":{"room_name":"会议室1203","room_limit":6},"4":{"room_name":"会议室1204","room_limit":6},"5":{"room_name":"会议1205","room_limit":18},"6":{"room_name":"休闲区12楼","room_limit":9},"7":{"room_name":"会议室1301","room_limit":6},"8":{"room_name":"会议室1302","room_limit":4},"9":{"room_name":"会议室1303","room_limit":6},"10":{"room_name":"会议室1304","room_limit":12},"11":{"room_name":"休闲区13楼","room_limit": 9 },"12":{"room_name":"贵宾室","room_limit":8}}
        
# https://oa.addcn.com/Home/Booked/delete
# booked_id=59306
# {state: "1", msg: "删除成功"}
# 取消会议室
functions.append({
    "name": "cancel_meetingroom_booking",
    "description": "cancel meetingroom booking",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "booking room name",
            },
            "date": {
                "type": "string",
                "description": "booking date, default is today, format is YYYY-MM-DD",
            },
            "at": {
                "type": "string",
                "description": "booking time",
            }
        },
        "required": ["at","name"],
    }
})
@print_result
async def cancel_meetingroom_booking(name, at, date=current_time(), user=None):
    macther = re.compile(r'.*(\d{4}|贵宾|休闲).*')
    _name = macther.match(name).group(1) if macther.match(name) is not None else name
    
    data, cookies = await get_meetingroom_info(date)
    booking = [booking for booking in data if macther.match(booking['room_name']).group(1) == _name and booking['start'] == at]
    if len(booking) == 0:
        return '抱歉，没有找到符合条件的会议室预定'
    booked_id = booking[0]['id']
    response = requests.post('https://oa.addcn.com/Home/Booked/delete', cookies=cookies, data={
        "booked_id": booked_id
    })
    _json = response.json()
    return {
        'message':_json['msg']
    }
    
functions.append({
    "name": "get_meetingroom_booking",
    "description": "get meetingroom booking",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "booking date, default is today, format is YYYY-MM-DD",
            }
        },
        "required": [],
    }
})
@print_result
async def get_meetingroom_booking(date=current_time(), user=None):
    booking_date = date[:10]
    bookings = await get_meetingroom_info(booking_date)
    setting_meetingrooms = get_room()
    setting_booking_timeslots = get_time_slot()
    
    for room in setting_meetingrooms.values():
        room['bookings'] = [{'start':booking['start'], 'end':booking['end']} for booking in bookings if booking['room_name'] == room['room_name']]
        room['available_timeslots'] = [timeslot for timeslot in setting_booking_timeslots if not is_time_overlap(timeslot['start'], timeslot['end'], room['bookings'][0]['start'], room['bookings'][0]['end'])]
    
    return {
        
    }

'''
目前有时段09:00-21:00，时段按每半小时拆分，格式为[{"start":"09:00", "end":"09:30"}]， 写一个方法，传入一个时段的开始和结束时间，计算出与这个时段不重叠的时段列表
'''
def get_available_timeslots(timeslots, start, end):
    available_timeslots = []
    for timeslot in timeslots:
        if not is_time_overlap(start, end, timeslot['start'], timeslot['end']):
            available_timeslots.append(timeslot)
    return available_timeslots