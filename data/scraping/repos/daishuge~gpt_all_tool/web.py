import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
from openai import OpenAI

client=OpenAI()

def gpt_api(query,rol="你是一个概括助手,擅长概括给定内容, 你将得到网页内容, 你应该把内容概括成80个中文字符以内"):
    """
    使用 OpenAI 的 ChatCompletion 创建聊天响应。

    参数:
        - query: 用户的查询内容
        - max: 响应的最大令牌数
        - if_print: 控制是否在控制台打印每个响应片段
        - tem: 生成响应的温度（创造性）
        - history: 对话历史记录(列表)
        - rol: 角色信息(字符串)
    """

    messages = [
        {"role": "system", "content": rol},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model='gpt-4-1106-preview',
        # model='gpt-4',
        messages=messages,
        temperature=0.75,
        max_tokens=1500,
        stream=True
    )

    result = ""

    for part in response:
        content = part.choices[0].delta.content if part.choices[0].delta and part.choices[0].delta.content else ""
        result += content
        if True:
            print(content, end='', flush=True)
            time.sleep(0.05)


    print("\n\n")
    return result

def get_url(query): # 传入 搜索关键词, 返回 url
    search_results = search(query, num_results=1, lang="en")
    
    # 获取生成器的第一个结果并返回
    first_result = next(search_results, None)
    return first_result

def gethtml(url):   # 传入 url, 返回 html
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查是否请求成功
        html = response.text
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除脚本和样式标签
        for script in soup(['script', 'style']):
            script.extract()
        
        # 获取纯文本内容
        text_content = soup.get_text()
        
        # 去除首尾的空白字符和空行
        lines = (line.strip() for line in text_content.splitlines())
        non_empty_lines = (line for line in lines if line)  # 过滤掉空行
        cleaned_content = '\n'.join(non_empty_lines)
        
        return cleaned_content
    except requests.exceptions.RequestException as e:
        print("请求出现问题:", e)
        return None

def main(reply, act):
    if act == "get_url_content":
        html = gethtml(reply)
        return html

    elif act == "google_search":
        url = get_url(reply)
        html = gethtml(url)
    
    reply=gpt_api(html)
    return reply