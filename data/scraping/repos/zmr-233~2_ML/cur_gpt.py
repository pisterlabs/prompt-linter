import os
import openai

# 设置API基础地址为代理服务器地址
openai.api_base = os.getenv("OPENAI_API_BASE")

# 从环境变量或其他安全的地方加载您的API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt(prompt):
    try:
        # 创建聊天请求
        response = openai.ChatCompletion.create(
            model="gpt-4-32k",  # 假设gpt-4是模型的标识符
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # 提取并返回GPT的回应
        gpt_response = response['choices'][0]['message']['content']
        return gpt_response
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        gpt_response = chat_with_gpt(user_input)
        print(f"GPT: {gpt_response}")
