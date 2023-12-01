# This is a sample Python script.
from token_counter import count_string_tokens
from bs4 import BeautifulSoup
from claude_api import Client
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_openai(prompt: str, model="gpt-3.5-turbo-16k-0613"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0
    )
    try:
        content = response['choices'][0]['message']['content']
        return content
    except Exception as e:
        print(e)

def ask_claude(prompt: str, attachment):
    cookie = os.getenv('CLAUDE_COOKIE')
    claude_api = Client(cookie)
    conversation_id = claude_api.create_new_chat()['uuid']
    response = claude_api.send_message(prompt, conversation_id, attachment=attachment)
    return response

def main_openai():
    html = ''

    soup = BeautifulSoup(html, "html.parser")

    for script in soup(["script", "style"]):
        script.decompose()
    # 10285
    count = count_string_tokens(str(soup), 'gpt-3.5-turbo-16k-0613')
    prompt = f'''你是一个写playwright的专家，下面是一个html，在---start之后，在---end之前，
    ---start
    {html}
    ---end
    请你阅读这个html，然后告诉我一些测试用例，越多越好'''
    response = ask_openai(prompt)
    print(response)

def main_claude():
    # 本来想加上6. 最后请严格遵守我的输入格式，我期望的输出格式是只需要告诉我文件代码即可，不需要返回其它任何字符
    # 但是反而效果变差了
    response = ask_claude("""你是一个写playwright的专家，下面是一个html，在html.txt中，请你阅读这个html，请你根据这个html，对“点击清除按钮,验证能清空当前输入”这个测试用例，编写一个playwright的测试代码。注意
1. nth的序号是用0开始的
2. 如果要测试输入，请仔细阅读html，然后在编写locator的时候尽可能详细地找到可以输入的dom对象
3. 如果要测试输入，请用type而不是fill
4. 如果要测试文本判断，请仔细阅读html，然后在编写locator的时候尽可能详细地找到最准确能够获取到对应文本的dom对象
5. 在使用locator来获取dom对象时，能不用div的标签选择器就不用div的标签选择器
""", './html.txt')
    print(response)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_claude()

