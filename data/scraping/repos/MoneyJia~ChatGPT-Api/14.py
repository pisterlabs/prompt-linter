import openai
import os
import pandas as pd  # 导入pandas库
import json  # 导入json模块

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

df_complex = pd.DataFrame({
    'Name': ['Alice','Bob', 'Charlie'],
    'Age' :[25 ,30,35] ,
    'Salary': [50000.0, 100000.5, 150000.75],
    'IsMarried': [True, False, True]
})
df_complex_json = df_complex.to_json(orient='split')

def calculate_total_age_from_split_json(input_json):

    df = pd.read_json(input_json, orient='split')

    total_age = df['Age'] .sum()

    return json.dumps({"total_age": str(total_age)})

print("df_complex_json:",df_complex_json)

result = calculate_total_age_from_split_json(df_complex_json)
print("The JSON output is:",result)

calculate_function_info = {
    "name": "calculate_total_age_from_split_json",
    "description": "计算年龄总和函数",
    "parameters": {
        "type": "object",
        "properties": {
            "input_json": {
                "type": "string",
                "description": "年龄数据集"
            }
        },
        "required": [
            "input_json"
        ]
    },
    "return": {
        "type": "string",
        "description": "年龄总和"
    },
}

function_repository = {
    "calculate_total_age_from_split_json": calculate_total_age_from_split_json,
}

functions = [calculate_function_info]

print(functions)

messages = [
    # GPT角色设定
    {"role": "system", "content": "你是一个优秀的数据分析师，现在有这样一份数据集：'%s'" % df_complex_json},
    # 模拟用户输入信息
    {"role": "user", "content": "请在数据集input_json上执行计算所有人年龄总和函数"}
]

complection = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=messages,
    functions=functions,
    function_call="auto"
)
print(complection)

function_name = complection.choices[0].message.function_call.name
function_args = json.loads(complection.choices[0]["message"]["function_call"]["arguments"])

print("function_name",function_name)
print("function_args",function_args)

input_json = function_args["input_json"]
print("input_json",input_json)


local_fuction_call = function_repository[function_name]

print("local_fuction_call",local_fuction_call)

final_response = local_fuction_call(input_json)

# print("final_response",final_response)

messages.append(complection.choices[0].message)

# 加function计算结果，注意: function message必须要输入关键词name
messages.append({"role": "function", "name": function_name, "content": final_response})


print("messages",messages)

lastResponse = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=messages
)

print("lastResponse",lastResponse)
print(lastResponse.choices[0].message["content"])
