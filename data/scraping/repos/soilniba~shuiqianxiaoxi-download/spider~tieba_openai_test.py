from bs4 import BeautifulSoup
import urllib
import json
import time
import datetime
import requests
import os
import re
import gzip
import openai
from config import openai_api_key, feishu_robot_news, feishu_robot_error

# 获取脚本所在目录的路径
script_dir = os.path.dirname(os.path.realpath(__file__))

# 切换工作目录到脚本所在目录
os.chdir(script_dir)

openai.api_key = openai_api_key
Cookie = ''
user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'
headers = {
    'User-Agent': user_agent, 
    'Connection': 'close',
    'Cookie': Cookie,
    'Accept-Encoding': 'gzip',
}
# proxy_handler = urllib.request.ProxyHandler({'socks5': '127.0.0.1:1080'})
# proxy_handler = urllib.request.ProxyHandler({'socks5': 'k814.kdltps.com:20818'})
socks5_proxies = 'socks5://t17842936906948:8z10sobl@l854.kdltps.com:20818'
# socks5_proxies = 'socks5://127.0.0.1:1080'
proxies = {
    'http': socks5_proxies,
    'https': socks5_proxies,
}
proxies = None

update_num = 0
add_num = 0
def get_news():
    global update_num, add_num
    update_num = 0
    add_num = 0
    file_name = 'news_gov.json'
    json_all = load_json(file_name)
    # clear_history_data(json_all)
    new_news_list = []
    if thread_list := get_list():
        get_page(thread_list, json_all, new_news_list)
        print("----新闻读取完毕----")
    else:
        print("thread_list读取失败")
        send_error_msg('出错啦！抓不到新闻啦！')
    print(f'新闻新增{add_num}条')
    write_json(file_name, json_all)
    if new_news_list:
        for data_info in new_news_list:
            href = data_info["href"]
            text = get_article(href)
            answer = ask_gpt(text)
            data_info['description'] = answer
            json_all[href] = data_info
            write_json(file_name, json_all)
            send_news(data_info)

def send_news(data_info):
    feishu_msg = {"content": []}
    # feishu_msg["title"] = '刚刚收到的新消息：'
    feishu_msg["content"].append([
        {
            "tag": "a",
            "text": data_info['title'],
            "href": f'{data_info["url"]}'
        },
        {
            "tag": "text",
            "text": '\n\n'
        },
    ])
    feishu_msg["content"].append([
        {
            "tag": "text",
            "text": data_info['description']
        },
    ])
    send_feishu_robot(FEISHU_ROBOT_ERROR, feishu_msg)

def send_error_msg(text):
    error_file_name = 'last_send_time_error.log'
    last_send_time = read_last_time(error_file_name)
    if time.time() - last_send_time > 1:  #报错间隔时间
        text_msg = text
        feishu_msg = {"content": []}
        feishu_msg["content"].append([
            {
                "tag": "text",
                "text": text_msg
            },
        ])
        send_feishu_robot(FEISHU_ROBOT_ERROR, feishu_msg)
        write_last_time(error_file_name)

def get_article(url = ''):
    # url = f'http://www.gov.cn{href}'
    # url = 'http://www.gov.cn/xinwen/2023-03/17/content_5747299.htm'
    # url = 'http://www.gov.cn/zhengce/zhengceku/2023-03/17/content_5747143.htm'
    # url = 'http://www.gov.cn/zhengce/zhengceku/2023-03/16/content_5746998.htm'
    # url = 'https://tieba.baidu.com/p/8312746395'
    response = requests.get(url)
    html = response.content
    # 解析网页内容
    soup = BeautifulSoup(html, 'html.parser')
    # 提取网页正文
    text = soup.get_text()
    # 去除多余空格、换行符等无用字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 将多个连续空格替换为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 输出处理后的文本
    # print(url, text)
    return text, soup.title.string

def ask_gpt(text):
    print(len(text))
    max_len = 3000
    if len(text) > max_len:
        text = text[:max_len]
    # 设置要发送到API的提示语
    prompt = f"请对以下新闻文章进行概述：\n{text}"
    message = []
    message.append({'role': 'system', 'content': '请对以下这篇文章标注关键词（不超过5个），然后引用一些重点语句(按权重从高到低排序，并在行首标出权重分数)'})
    message.append({'role': 'user', 'content': text})
    

    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0301",  # 对话模型的名称
            # model = "gpt-4-0314",  # 对话模型的名称
            messages = message,
            temperature = 0.9,  # 值在[0,1]之间，越大表示回复越具有不确定性
            # max_tokens=4097,  # 回复最大的字符数
            top_p = 1,
            frequency_penalty = 0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty = 0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
        )
        print("[ChatGPT] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))
        return response.choices[0]['message']['content']
    except Exception as e:
        print(e)
        send_error_msg(f'openai api error:{e.user_message}')
    # except openai.error.RateLimitError as e:
    #     # rate limit exception
    #     print(e)
    #     if retry_count < 1:
    #         time.sleep(5)
    #         logger.warn("[OPEN_AI] RateLimit exceed, 第{}次重试".format(retry_count+1))
    #         return self.reply_text(session, session_id, retry_count+1)
    #     else:
    #         return {"completion_tokens": 0, "content": "提问太快啦，请休息一下再问我吧"}
    # except openai.error.APIConnectionError as e:
    #     # api connection exception
    #     logger.warn(e)
    #     logger.warn("[OPEN_AI] APIConnection failed")
    #     return {"completion_tokens": 0, "content":"我连接不到你的网络"}
    # except openai.error.Timeout as e:
    #     logger.warn(e)
    #     logger.warn("[OPEN_AI] Timeout")
    #     return {"completion_tokens": 0, "content":"我没有收到你的消息"}
    # except Exception as e:
    #     # unknown exception
    #     logger.exception(e)
    #     Session.clear_session(session_id)
    #     return {"completion_tokens": 0, "content": "请再问我一次吧"}

def get_html(url):
        url = urllib.parse.quote(url, safe='/:?=&')
        # request = urllib.request.Request(url, headers = headers)
        # response = urllib.request.urlopen(request)
        if proxies:
            response = requests.get(url, headers=headers, proxies=proxies)
        else:
            response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        HtmlContent = response.read() if hasattr(response, 'read') else response.text
        # HtmlContent = HtmlContent.decode('utf-8')
        # print('python 返回 URL:{} 数据成功'.format(url))
        return HtmlContent

def get_list():  # 获取单页JSON数据
    url = "http://www.gov.cn/xinwen/lianbo/bumen.htm"
    HtmlContent = get_html(url)
    HtmlContent = HtmlContent.replace("<!--", "")
    HtmlContent = HtmlContent.replace("-->", "")
    soup = BeautifulSoup(HtmlContent, "lxml")
    thread_list = soup.select_one('body > div.main > div > div > div.news_box > div')
    # print(thread_list)
    return thread_list

def get_page(thread_list, json_all, new_news_list):
    li_list = thread_list.select('li')
    for li in li_list:
        a = li.select_one('a')
        title = a.text
        href = a.attrs['href']
        span = li.select_one('span')
        date = span.text.strip()
        # print(title, href, date)
        if href in json_all:
            data_info = json_all[href]
            if 'href' not in data_info:
                data_info['href'] = href
        else:
            data_info = {}
            data_info['href'] = href
            data_info['title'] = title
            data_info['date'] = date
            json_all[href] = data_info
            # new_news_list.append(data_info)
            new_news_list.insert(0, data_info)
            global add_num
            add_num += 1

def write_json(file_name, json_all):
    str_json = json.dumps(json_all, indent=2, ensure_ascii=False)
    with open(file_name, "w", encoding='utf-8') as f:
        f.write(str_json)
        f.close()

def load_json(file_name):
    try:
        f = open(file_name, "r", encoding='utf-8')
    except IOError:
        return {}
    else:
        return json.load(f)

def send_wx_robot(robot_url, content_msg, mentioned_list = None):
    headers = {
        'Content-Type': 'application/json',
    }
    if mentioned_list:
        data_table = {
            "msgtype": "text", 
            "text": { "content": content_msg, "mentioned_list": mentioned_list }
        }
    else:
        data_table = {
            "msgtype": "markdown", 
            "markdown": { "content": content_msg }
        }
    data = json.dumps(data_table)
    response = requests.post(f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={robot_url}', headers=headers, data=data)

def send_feishu_robot(feishu_robot_key, feishu_msg):
    headers = {
        'Content-Type': 'application/json',
    }
    data = json.dumps({
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": feishu_msg
            }
        }
    })
    response = requests.post(f'https://open.feishu.cn/open-apis/bot/v2/hook/{feishu_robot_key}', headers=headers, data=data)

def get_feishu_token():
    headers = {
        'Content-Type': 'application/json',
    }
    data = json.dumps({
        "app_id": "cli_a1c3790e21f8100c",
        "app_secret": "YVXgZL2HnYi6gHm2NmxenfOTi60rfrQ3",
    })
    response = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal', headers=headers, data=data)
    responsejson = json.loads(response.text)
    print(responsejson['tenant_access_token'])
    return responsejson['tenant_access_token']

def GetUserIDs(email_list):
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': 'Bearer ' + get_feishu_token(),
    }
    response = requests.post('https://open.feishu.cn/open-apis/user/v1/batch_get_id?emails=' + '&emails='.join(email_list), headers=headers)
    responsejson = json.loads(response.text)
    email_users = responsejson['data']['email_users']
    user_id_list = []
    for email, ids in email_users.items():
        print(email, ids[0]['open_id'], ids[0]['user_id'])
        user_id_list.append(ids[0]['user_id'])
    return user_id_list

def write_last_time(file_name):
    with open(file_name, "w") as f:
        f.write(str(time.time()))
        f.close()

def read_last_time(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            last_send_time = float(f.read())
            f.close()
            return last_send_time
    else:
        write_last_time(file_name)
        return time.time()

def main():
    lock_file = 'news_spider.lock'
    if not os.path.exists(lock_file):
        _extracted_from_main_4(lock_file)
    else:
        print('file lock')
        time.sleep(5)
        os.remove(lock_file)
        print('lock file delete')

def _extracted_from_main_4(lock_file):
    # with open(lock_file, 'w') as f:
    #     f.write('')
    #     f.close()
    get_news()
    if os.path.exists(lock_file):
        os.remove(lock_file)

def check_local_ip():
    url = 'https://www.123cha.com'
    HtmlContent = get_html(url)
    soup = BeautifulSoup(HtmlContent, "lxml")
    iplocation = soup.select_one('body > div.header > div.location > span')
    print('当前访问IP:', iplocation and iplocation.text)

if __name__ == "__main__":
    try:
        # 可能会引发异常的代码
        check_local_ip()
    except Exception as e:
        # 处理异常的代码
        print('Error:', e)
        result = None
    # main()
    url = 'https://tieba.baidu.com/p/8312711920'
    text, title = get_article(url)
    answer = ask_gpt(text)
    send_news({
        'title': title.replace("【图片】", "").replace("_百度贴吧", ""), 
        'url': url,
        'description': answer,
    })
