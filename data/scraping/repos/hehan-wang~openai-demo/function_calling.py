import openai
import json

openai.api_key = 'sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327'
openai.api_base = "https://key.wenwen-ai.com/v1"

# 1. 定义函数
# 1.1 定义模拟获取天气的本地函数
def get_current_weather(location, unit):
    # Call the weather API
    return f"It's 20 {unit} in {location}"


# 1.2 定义函数字典方便调用
function_dict = {
    "get_current_weather": get_current_weather,
}

# 1.3 定义chat接口需要的函数
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

# 2. 第一次调用chat接口，返回的是函数调用的提示
messages = [
    {"role": "user", "content": "What's the weather like in Boston today with celsius?"}]
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto",  # auto is default, but we'll be explicit
)

# {
#   "id": "chatcmpl-8DxAr4kjYpJpf6oPoxp8xDgeTP90a",
#   "object": "chat.completion",
#   "created": 1698336421,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": null,
#         "function_call": {
#           "name": "get_current_weather",
#           "arguments": "{\n  \"location\": \"Boston, MA\"\n}"
#         }
#       },
#       "finish_reason": "function_call"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 83,
#     "completion_tokens": 18,
#     "total_tokens": 101
#   }
# }
print(completion)

# 3. 从结果接口的结果中获取函数调用的参数 进行本地函数调用
# 3.1 获取函数调用的参数
response_message = completion.choices[0].message
function_name = response_message["function_call"]["name"]
function_args = json.loads(response_message["function_call"]["arguments"])
# 3.2 调用本地函数
function_response = function_dict.get(function_name)(**function_args)
# 3.3 将本地函数的结果作为chat接口的输入
messages.append(response_message)
messages.append({
    "role": "function",
    "name": function_name,
    "content": function_response,
})

# 4. 第二次调用chat接口，返回的是chat的最终结果
completion_final = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)

print(completion_final.choices[0].message)