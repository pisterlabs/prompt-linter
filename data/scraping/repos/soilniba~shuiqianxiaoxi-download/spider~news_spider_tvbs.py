from bs4 import BeautifulSoup
import urllib
import json
import time
import datetime
import requests
import os
import re
import opencc
import traceback
import gzip
import PyPDF2
import docx2txt
import nltk
import html2text
import openai
from loguru import logger
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTKeywordTableIndex,
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    BeautifulSoupWebReader,
    StringIterableReader,
    LLMPredictor,
    PromptHelper,
    QuestionAnswerPrompt,
    RefinePrompt,
    ServiceContext
)
from config import openai_api_key, feishu_robot_tvbs, feishu_robot_error

script_dir = os.path.dirname(os.path.realpath(__file__))    # 获取脚本所在目录的路径
os.chdir(script_dir)                                        # 切换工作目录到脚本所在目录
filename_ext = os.path.basename(__file__)
file_name, file_ext = os.path.splitext(filename_ext)
logger.add(f"{file_name}.log", format="{time} - {level} - {message}", rotation="10 MB", compression="zip")    # 添加日志文件
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
import psutil
p = psutil.Process()  # 获取当前进程的Process对象
p.nice(psutil.IDLE_PRIORITY_CLASS)  # 设置进程为低优先级
# feishu_robot_tvbs = feishu_robot_error  # 强制使用测试频道
converter = opencc.OpenCC('tw2sp.json')  # 创建转换器对象， 繁體（臺灣正體標準）到簡體並轉換爲中國大陸常用詞彙

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
    file_name = 'news_tbvs.json'
    json_all = load_json(file_name)
    # clear_history_data(json_all)
    new_news_list = []
    try:
        thread_list_all = get_list_all()
        for thread_list in thread_list_all:
            get_page(thread_list['thread_list'], thread_list['category'], json_all, new_news_list)
        print("----新闻读取完毕----")
    except Exception as e:
        send_error_msg(f'出错啦！tbvs抓不到新闻啦！\n{e}')
    print(f'新闻新增{add_num}条')
    write_json(file_name, json_all)
    for href, data_info in reversed(json_all.items()):
        if not data_info.get('send_time'):
            if not data_info.get('description'):
                try:
                    href = data_info["href"]
                    # text = get_article(href)
                    # answer = ask_gpt(text)
                    answer = ask_llama_index(href)
                    if answer is None:
                        answer = 'None'
                    data_info['description'] = answer
                    json_all[href] = data_info
                    write_json(file_name, json_all)
                except Exception as e:
                    # tb_str = traceback.format_exc(limit=3)
                    send_error_msg(f'ask_llama_index error\n{e}')
                    continue
            if data_info.get('description') and data_info.get('description') != 'None':
                # data_info['send_time'] = None
                data_info['send_time'] = time.time()
                write_json(file_name, json_all)
                send_news(data_info)

def send_news(data_info):
    feishu_msg = {"content": []}
    # feishu_msg["title"] = '刚刚收到的新消息：'
    feishu_msg["content"].append([
        {
            "tag": "text",
            "text": f"[{data_info['category']}]"
        },
        {
            "tag": "a",
            "text": converter.convert(data_info['title']),
            "href": f'https://news.tvbs.com.tw{data_info["href"]}'
        },
        {
            "tag": "text",
            "text": f"{data_info['date']}"
        },
    ])
    if data_info.get('description'):
        feishu_msg["content"].append([
            {
                "tag": "text",
                "text": data_info.get('description')
            },
        ])
    send_feishu_robot(feishu_robot_tvbs, feishu_msg)

def send_error_msg(text):
    if feishu_robot_error:
        text_msg = text
        feishu_msg = {"content": []}
        feishu_msg["content"].append([
            {
                "tag": "text",
                "text": text_msg
            },
        ])
        send_feishu_robot(feishu_robot_error, feishu_msg)
    logger.error(text)

def get_article(url):
    response = requests.get(url)
    html = response.content
    # 解析网页内容
    soup = BeautifulSoup(html, 'html.parser')
    div_main = soup.select_one('#news_detail_div')
    # 去除广告
    if div_guangxuan := div_main.select_one('div.guangxuan'):
        div_guangxuan.extract()
    # 提取网页正文
    text = div_main.get_text()
    # 去除多余空格、换行符等无用字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 将多个连续空格替换为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 输出处理后的文本
    # print(url, text)
    return text

def ask_llama_index(href):
    # define LLM
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=2048))
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0))

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    chunk_size_limit = 10000
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # doc是你文档所存放的位置，recursive代表递归获取里面所有文档
    # documents = SimpleDirectoryReader(input_dir=os.path.dirname(__file__) + '/doc',recursive=True).load_data()
    url = f'https://news.tvbs.com.tw{href}'
    documents = StringIterableReader().load_data(texts=[get_article(url)])
    for doc in documents:
        doc.text = doc.text.replace("。", ". ")
    # documents = BeautifulSoupWebReader().load_data([url])
    # index = GPTSimpleVectorIndex.from_documents(documents)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    # index = GPTSimpleVectorIndex.from_documents(documents)
    # save_json_path = os.path.dirname(__file__) + '\\index.json'
    # index.save_to_disk(save_json_path);

    # query_index.py   从index文件里获得相关资料并向GPT提问
    # index = GPTKeywordTableIndex.load_from_disk(save_json_path, service_context=service_context)
    # Context information is below. 
    # ---------------------
    # {context_str}
    # ---------------------
    # Given the context information and not prior knowledge, answer the question: {query_str}
    text_qa_prompt_tmpl = (
        "我们在下面提供了上下文信息. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "鉴于此信息，请回答以下问题: {query_str}\n"
    )
    # The original question is as follows: {query_str}
    # We have provided an existing answer: {existing_answer}
    # We have the opportunity to refine the existing answer (only if needed) with some more context below.
    # ------------
    # {context_msg}
    # ------------
    # Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.
    refine_prompt_tmpl = (
        "之前我们询问过这个问题: {query_str}\n"
        "得到了原始的答案: {existing_answer}\n"
        "现在我们有机会完善现有的答案 (仅在需要时) 通过下面的更多上下文.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "给我一个新的答案, 完善原始答案以更好的回答问题. 如果新的上下文没有用或者没必要再完善了, 则重复一遍原始的答案.\n"
    )
    text_qa_prompt = QuestionAnswerPrompt(text_qa_prompt_tmpl)
    refine_prompt = RefinePrompt(refine_prompt_tmpl)

    # while True:
    #     ask = input("请输入你的问题：")
    #     print(index.query(ask))
    answer = index.query("用中文总结一下这篇文章主要讲了啥", 
                         text_qa_template = text_qa_prompt,
                         refine_template = refine_prompt)
    time.sleep(10)
    return answer.response

def ask_gpt(text):
    print(len(text))
    max_len = 3000
    if len(text) > max_len:
        text = text[:max_len]
    # 设置要发送到API的提示语
    prompt = f"请对以下新闻文章进行概述：\n{text}"
    message = [
        {'role': 'system', 'content': '请用中文对以下新闻文章进行概述'},
        {'role': 'user', 'content': text},
    ]
    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",  # 对话模型的名称
            # model = "gpt-4",  # 对话模型的名称
            messages = message,
            temperature = 0.9,  # 值在[0,1]之间，越大表示回复越具有不确定性
            #max_tokens=4096,  # 回复最大的字符数
            top_p = 1,
            frequency_penalty = 0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty = 0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
        )
        print(
            f"""[ChatGPT] reply={response.choices[0]['message']['content']}, total_tokens={response["usage"]["total_tokens"]}"""
        )
        return response.choices[0]['message']['content']
    except Exception as e:
        print(e)
        send_error_msg(f'openai api error:{e}')


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

def get_list_all():
    thread_list_all = [
        {
            'thread_list': get_list('https://news.tvbs.com.tw/realtime/china'),
            'category': '大陆',
        },
        # {
        #     'thread_list': get_list('https://news.tvbs.com.tw/realtime/world'),
        #     'category': '全球',
        # },
        {
            'thread_list': get_list('https://news.tvbs.com.tw/realtime/tech'),
            'category': '科技',
        },
    ]
    return thread_list_all

def get_list(url):  # 获取单页JSON数据
    HtmlContent = get_html(url)
    HtmlContent = HtmlContent.replace("<!--", "")
    HtmlContent = HtmlContent.replace("-->", "")
    soup = BeautifulSoup(HtmlContent, "lxml")
    thread_list = soup.select_one('body > div.container > main > div > article > div.news_list > div.list')
    # print(thread_list)
    return thread_list

def get_page(thread_list, category, json_all, new_news_list):
    li_list = thread_list.select('li')
    for li in li_list:
        a = li.select_one('a')
        if a is not None:
            title = a.text
            href = a.attrs['href']
            date_div = li.select_one('div[class="time"]')
            date = date_div.text.strip() if date_div is not None else ""
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
                data_info['category'] = category
                json_all[href] = data_info
                # new_news_list.append(data_info)
                new_news_list.insert(0, data_info)
                global add_num
                add_num += 1
                if add_num > 10:
                    # 只读前十条，太旧的就不看了
                    break
            # if data_info['href'] == '/zhengce/zhengceku/2023-03/15/content_5746847.htm':
            #     new_news_list.append(data_info)


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
    return json.loads(response.text)

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
    main()