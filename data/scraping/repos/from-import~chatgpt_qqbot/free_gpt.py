import openai

openai.api_key = "sk-N6xwkHWGIZx0VyMw4yoTN0lqxAGEJ2zgGT4NGxTk3fLCQgF1"
openai.api_base = "https://api.chatanywhere.com.cn/v1"

def chatgpt(message):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                              messages=[{"role": "user", "content": message}])
    return completion.choices[0].message.content

# print(chatgpt("你好"))

while(1):
	print(chatgpt(input("input:")))
