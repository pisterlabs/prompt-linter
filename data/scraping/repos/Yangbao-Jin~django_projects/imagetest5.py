import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')
import openai

# 加载你的OpenAI API密钥
#openai.api_key = 'your-api-key'

# 初始化会话状态
conversation_history = []

# 定义发送请求的函数
def send_message(message_text, user_role='user'):
    # 将用户的消息添加到会话历史中
    conversation_history.append({'role': user_role, 'content': message_text})
    
    # 发送请求到OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview", # 或者你选择的其它模型
        messages=conversation_history
    )
    
    # 提取模型的回复并添加到会话历史中
    message = response['choices'][0]['message']['content']
    conversation_history.append({'role': 'assistant', 'content': message})
    
    # 返回模型的回复
    return message

# 一个简单的对话循环示例
while True:
    # 获取用户输入
    user_input = input("You: ")
    
    # 检查是否结束对话
    if user_input.lower() == 'quit':
        break
    
    # 发送消息并获取回复
    reply = send_message(user_input)
    
    # 打印模型的回复
    print(f"Assistant: {reply}")
