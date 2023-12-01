import openai
import argparse
import sys
import os
import subprocess
from termcolor import colored

# OpenAI API 访问令牌
openai.api_key = "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"

# 聊天记录缓存
chat_log = []

# 模型默认值
default_model = "text-davinci-002"
default_max_tokens = 1024

# 解析命令行参数
parser = argparse.ArgumentParser(description="Chat with GPT")
parser.add_argument("-m", "--model", type=str,
                    default=default_model, help="GPT模型名称")
parser.add_argument("-t", "--max_tokens", type=int,
                    default=default_max_tokens, help="最大输出字符限制")
args = parser.parse_args()

# 初始化 OpenAI


def init():
    try:
        models = openai.Model.list()
        model_names = [model.name for model in models['data']]
        if args.model not in model_names:
            print(f"模型 {args.model} 不存在，请输入有效的模型名称.")
            sys.exit()
    except Exception as e:
        print("OpenAI 认证失败，请检查 API 访问令牌是否正确.")
        sys.exit()

# 与 OpenAI API 进行交互


def call_openai_api(prompt, model, max_tokens):
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5
        )
        message = response.choices[0].text.strip()
        return message
    except Exception as e:
        print(f"OpenAI API 访问失败：{e}")
        sys.exit()

# 将聊天记录保存到文件


def save_chat_log(filename):
    try:
        with open(filename, "w") as f:
            for message in chat_log:
                f.write(f"{message}\n")
        print(f"聊天记录已保存到文件 {filename}.")
    except Exception as e:
        print(f"无法保存聊天记录到文件 {filename}：{e}")

# 加载聊天记录文件


def load_chat_log(filename):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                chat_log.append(line.strip())
        print(f"已从文件 {filename} 中加载聊天记录.")
    except Exception as e:
        print(f"无法加载聊天记录文件 {filename}：{e}")

# 执行 Linux 命令


def execute_command(command):
    try:
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        return output
    except Exception as e:
        return f"执行命令时出错：{e}"

# 主程序循环


def main_loop():
    global chat_log
    while True:
        # 获取用户输入
        try:
            user_input = input("> ")
        except KeyboardInterrupt:
            print("\n退出程序.")
            sys.exit()
        # 处理命令
        if user_input.startswith("input "):
            filename = user_input[6:]
            try:
                with open(filename, "r") as f:
                    prompt = f.read().strip
            except Exception as e:
                print(f"无法打开文件 {filename}：{e}")
                continue
        elif user_input.startswith("out "):
            filename = user_input[4:]
            save_chat_log(filename)
            continue
        elif user_input.startswith("!"):
            command = user_input[1:]
            output = execute_command(command)
            print(output)
            continue
        else:
            prompt = user_input
    # 与 OpenAI API 进行交互
        message = call_openai_api(prompt, args.model, args.max_tokens)
        chat_log.append(f"{prompt} => {message}")
    # 输出响应
        print(colored(message, "green"))
if __name__=='__main__':
    main_loop()

