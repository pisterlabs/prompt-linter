import base64
import openai
import requests
import json
from Crypto.Cipher import AES  # pip install pycryptodome

openai_url = 'https://api.openai.com/v1/completions'
sbaliyun_url = 'https://cc-api.sbaliyun.com/v1/completions'
sbaliyun_headers = {
    'Content-Type': 'application/json',
    'referer': 'https://chatgpt.sbaliyun.com/',
}

def sbaliyun():
    while True:
        msg = input("请输入要问的话：")
        if msg == 'exit':
            break
        msg = aes_encrypt(msg)
        json_res = json.dumps({"prompt": msg})
        result = requests.post(url=sbaliyun_url, headers=sbaliyun_headers, data=json_res)
        try:
            print(json.loads(result.text)["choices"][0]["text"])
            print()
        except:
            print("请重新请求")


def aes_encrypt(text):
    key = 'L#$@XowPu!uZ&c%u'.encode('utf-8')
    iv = '2auvLZzxz7bo#^84'.encode('utf-8')
    padding = lambda data: data + (16 - len(data.encode('utf-8')) % 16) * chr(16 - len(data.encode('utf-8')) % 16)
    text = padding(text).encode("utf-8")
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    res = encryptor.encrypt(text)
    return base64.b64encode(res).decode('utf-8')

def set_key():
    while True:
        open_key = input("请输入您的key：")
        print("检测key是否正确...")
        # 检测key是否正确
        if open_key[:3] != 'sk-':
            print("key错误，请重新设置key")
            continue
        openai.api_key = open_key
        json_res = json.dumps({"model": "text-davinci-003", "prompt": "你", "max_tokens": 2048, "temperature": 0.5, "n": 1,
             "stream": False, "stop": "§"})
        openai_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + openai.api_key,
        }
        result = requests.post(url='https://api.openai.com/v1/completions', headers=openai_headers, data=json_res)
        # 判断key是否正确
        if result.status_code != 200:
            print("key错误，请重新设置key")
            continue
        print("设置成功,您的key为：", open_key)
        break

def req_api():
    set_key()
    while True:
        msg = input("请输入要问的话：")
        if msg == 'exit':
            break
        if msg == 'set_key':
            set_key()
            continue
        json_res = json.dumps({"model": "text-davinci-003","prompt": msg, "max_tokens": 2048, "temperature": 0.5, "n": 1, "stream": False, "stop": "§"})
        openai_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + openai.api_key,
        }
        result = requests.post(url=openai_url, headers=openai_headers, data=json_res)
        # 判断key是否正确
        if result.status_code == 401:
            print("key错误，请重新设置key")
            set_key()
            continue
        try:
            print(json.loads(result.text)["choices"][0]["text"])
            print()
        except:
            print("请重新请求")

if __name__ == '__main__':
    while True:
        print("ChatGPT - OpenAI".center(50, '-'))
        print("输入 'exit' 返回上一级或退出，输入 'set_key' 重新设置key")
        print("输入 '1' 使用自己的Key")
        print("输入 '2' 使用sbaliyun")
        num = input("请输入您的选择：")
        if num == 'exit':
            break
        if num == '1':
            req_api()
        elif num == '2':
            sbaliyun()
        else:
            print("输入错误，请重新输入")

