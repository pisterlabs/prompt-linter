import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key  = os.getenv('OPENAI_KEY')
openai_model = os.getenv('OPENAI_MODEL')

system = f"""
你是一個助手，可以根據給定的訊息生成文本，以形成人們易於理解的答案。
最新的提示包含訊息，你需要基於給定的訊息生成人類可讀的回應。
讓它的訊息回覆看起來像來自一個 AI 助手，但不要添加任何額外訊息。
不要添加任何額外的訊息，除非最新的提示中有明確提供。
我再次強調，不要添加任何未明確給定的訊息。
"""

def generate_response(messages):
    messages = [
        {"role": "system", "content": system}
    ] + messages
    print(messages)
    # Make a request to OpenAI
    completions = openai.ChatCompletion.create(
        model=openai_model,
        messages=messages,
        temperature=0.0
    )
    response = completions.choices[0].message.content
    print(response)
    # If the model apologized, remove the word
    if "抱歉" in response:
        response = " ".join(response[2:])
    return response


if __name__ == '__main__':
    data = [{"market":"上市","code":"2330","name":"台積電","pagerank":0.002635887482379148,"community":0,"stock_id":"2330"}]
    print(generate_response([{'role': 'user', 'content': str(data)}]))
