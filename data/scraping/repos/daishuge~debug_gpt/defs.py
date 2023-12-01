import subprocess
import re
import openai
import time

def terminal(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True, encoding='utf-8', errors='ignore')
        stdout = result.stdout
        stderr = result.stderr
        
        if stderr:
            print("ERROR?\n"+stderr)
            return stderr

        return stdout

    except Exception as e:
        return str(e)


# def get_code(md_str):   # code_list=get_code(md_str)
#     '''
#     传入md文本,返回代码块列表
#     调用方法:
#     code_list=get_code(md_str)
#     for item in code_list:
#         print(item)
#     '''
#     code_blocks = re.findall(r"```(?:[a-zA-Z]+\n)?([\s\S]*?)```", md_str)
    
#     code_blocks = [block.strip() for block in code_blocks]
    
#     return code_blocks

import re

def get_code(md_str):
    '''
    传入md文本,返回两种代码块列表：Python代码块和终端命令块
    调用方法:
    python_code_list, terminal_code_list = get_code(md_str)
    for code in python_code_list:
        print("Python Code:", code)
    for code in terminal_code_list:
        print("Terminal Command:", code)
    '''

    # 匹配Python代码块
    python_code_blocks = re.findall(r"```python\n(.*?)```", md_str, re.DOTALL)

    # 匹配终端命令块
    terminal_code_blocks = re.findall(r"```(bash|shell)\n(.*?)```", md_str, re.DOTALL)
    terminal_code_blocks = [block[1] for block in terminal_code_blocks]  # 从元组中取出命令部分

    # 移除多余的空格
    python_code_blocks = [block.strip() for block in python_code_blocks]
    terminal_code_blocks = [block.strip() for block in terminal_code_blocks]
    
    return python_code_blocks, terminal_code_blocks

  
def fake_api(query,mod,if_print):     # fake_api("hi! are you gpt?",True,1,"gpt-3.5-turbo")

    openai.api_key = "fk-yU3UOSY13E9WsAkKBTrkq1KiiqIsAXBs_J6OTh_yvJM"
    openai.api_base = "https://ai.fakeopen.com/v1/" 

    response = openai.ChatCompletion.create(
        model=mod,
        messages=[
            {'role': 'user', 'content': query}
        ],
        temperature=0.7,
        max_tokens=2000,
        stream=True
    )

    result = ""

    for chunk in response:
        # 确保字段存在
        if 'choices' in chunk and 'delta' in chunk['choices'][0]:
            chunk_msg = chunk['choices'][0]['delta'].get('content', '')
            result += chunk_msg  # 将输出内容附加到结果字符串上

            if if_print:
                print(chunk_msg, end='', flush=True)
                time.sleep(0.05)

    return result

def print_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'reset': '\033[0m'
    }
    
    print(colors[color] + text + colors['reset'])