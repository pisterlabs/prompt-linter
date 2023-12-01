import sys
import traceback
import openai
import os

script_name = os.path.basename(os.path.abspath(__file__))

# openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_key = ""

# 这里修改为你要执行的脚本名称
hook_file = 'Jump回归.py'


def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
        api_key (str): OpenAI API 密钥

    Returns:
        tuple: (results, error_desc)
    """
    try:
        print("程序出错了, 正在分析报错问题中, 请稍后...\n")
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                print(completion['content'])
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                completion[delta_k] += delta_v
        messages.append(completion)  # 直接在传入参数 messages 中追加消息
        return (True, '')
    except Exception as err:
        return (False, f'OpenAI API 异常: {err}')


def send_question_to_gpt(question):
    content_str = f"你现在将扮演一个资深的python工程师, 非常擅长处理程序报错时的问题, 我现在正在利用{script_name}脚本来捕获运行{hook_file}这个脚本时的报错, 报错如下: \n" + question + f"\n不要去定位python某个库的报错信息, 而是{hook_file}这个脚本的报错信息中指出报错的位置, 并告诉我该如何进行对应修改"
    messages = [{'role': 'user', 'content': content_str}, ]
    gpt_35_api(messages)


def excepthook(exc_type, exc_value, exc_traceback):
    # 将异常信息打印到控制台
    msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
    all_msg = "".join(msg)
    send_question_to_gpt(all_msg)


# 设置sys.excepthook为自定义函数
sys.excepthook = excepthook

# 运行原始代码的内容
exec(open(hook_file, encoding='utf-8').read())
