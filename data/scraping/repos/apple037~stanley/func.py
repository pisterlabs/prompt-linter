import base64
import json
import random
from io import BytesIO

import aiohttp
import openai
import replicate
import requests
from PIL import Image


def get_currency_price(name):
    f = open('../resource/mapping.json', encoding='utf-8')
    data = json.load(f)
    res = ""
    maps = data["mapping"]
    match = False
    for i in range(0, len(maps)):
        if name == maps[i]["name"] or in_object(name, maps[i]["alias"]):
            name = maps[i]["name"]
            match = True
    if not match:
        name = "all"
    r = requests.get('https://poe.ninja/api/data/CurrencyOverview?league=Scourge&type=Currency&language=en')
    j = r.json()
    lines = j["lines"]
    for x in range(0, len(lines)):
        if name == "all":
            c = str(lines[x]['currencyTypeName']) + ": " + str(lines[x]['chaosEquivalent'])
            res = res + "\n" + json.dumps(c)
        else:
            if lines[x]['currencyTypeName'] == name:
                res = lines[x]['chaosEquivalent']
    return res


def get_help():
    res = "#price(#f) {currency_name}: 查詢通貨價格"
    res = res + "\n" + "#item {item_name}: 查詢物品最多上架價格(還沒做)"
    return res


def in_object(name, maps):
    flag = False
    for i in range(0, len(maps)):
        if str(name) == maps[i]:
            flag = True
    return flag


def get_map():
    f = open('../resource/mapping.json', encoding='utf-8')
    data = json.load(f)
    res = ""
    maps = data["mapping"]
    for i in range(0, len(maps)):
        if i == 0:
            res = str(maps[i]["name"]) + " : " + str(maps[i]["alias"])
        else:
            res = res + "\n" + str(maps[i]["name"]) + " : " + str(maps[i]["alias"])
    return res


def get_answer(question, api_key):
    print("Receive question: " + question)
    openai.api_key = api_key
    temperature = random.randint(0, 100) / 100
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': '接收任何語言的問題並使用繁體中文回答'},
            {'role': 'user', 'content': question}
        ],
        stream=True
    )
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []

    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message

    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    print(f"Full conversation received: {full_reply_content}")
    print('Temperature: ' + str(temperature))
    # print('Generated answer: ' + full_reply_content)
    return full_reply_content


def get_answer_with_history(message_list, api_key):
    openai.api_key = api_key
    temperature = random.randint(0, 100) / 100
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=temperature,
        messages=message_list,
        stream=True
    )
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []

    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message

    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    print(f"Full conversation received: {full_reply_content}")
    print('Temperature: ' + str(temperature))
    # print('Generated answer: ' + full_reply_content)
    return full_reply_content


async def generate_image(prompts, sd_url):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    false = False
    true = True
    request_body = {
        "enable_hr": false,
        "denoising_strength": 0,
        "firstphase_width": 0,
        "firstphase_height": 0,
        "hr_scale": 2,
        "hr_upscaler": "",
        "hr_second_pass_steps": 0,
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "styles": [],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 30,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "restore_faces": true,
        "tiling": false,
        "do_not_save_samples": false,
        "do_not_save_grid": false,
        "negative_prompt": "bad-hands-5, EasyNegative, ng_deepnegative_v1_75t",
        "eta": 0,
        "s_churn": 0,
        "s_tmax": 0,
        "s_tmin": 0,
        "s_noise": 1,
        "override_settings": {},
        "override_settings_restore_afterwards": true,
        "script_args": [],
        "sampler_index": "DPM++ SDE Karras",
        "script_name": "",
        "send_images": true,
        "save_images": false,
        "alwayson_scripts": {},
        "prompt": prompts
    }
    txt2img_url = sd_url + "/sdapi/v1/txt2img"
    print('Prompt used: ' + prompts)
    async with aiohttp.request('POST', url=txt2img_url, headers=headers, data=json.dumps(request_body)) as response:
        # response = requests.post(txt2img_url, headers=headers, data=json.dumps(request_body))
        if response.status == 200:
            # print(response.status)
            chunk = await response.content.read()
            res = chunk.decode('utf-8')
            # print(json.loads(res))
            images = json.loads(res)['images'][0]
            byte_data = base64.b64decode(images)
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)
            return img


async def sd_info(sd_url):
    info_url = sd_url + "/sdapi/v1/progress"
    response = requests.get(info_url)
    if response.status_code == 200:
        progress = response.json()['progress']
        eta = response.json()['eta_relative']
        if progress == 0.0 and eta == 0.0:
            return "Stable diffusion server is free now"
        message = "Progress " + str(progress) + "% and eta is " + str(eta) + " seconds!"
        return message
    else:
        message = "Error when get progress"
        return message


async def img2txt(image_url, api_key):
    # 使用 Replicate 套件呼叫模型
    client = replicate.Client(api_token=api_key)
    result = client.run(
        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        input={"image": image_url}
    )
    print("Result: " + result)
    return result


async def random_cat():
    response = requests.get('https://cataas.com/cat')
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None
