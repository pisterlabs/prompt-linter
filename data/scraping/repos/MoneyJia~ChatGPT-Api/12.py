import openai
import os
import pandas as pd  # 导入pandas库

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

# df_complex = pd.DataFrame({
# 'Name': ['Alice','Bob', 'Charlie'],' Age' :[25 ,6 30,35] ,
# 'Salary': [50000.0, 100000.5, 150000.75]'IsMarried': [True, False, True]

df_complex = pd.DataFrame({
    'Name': ['Alice','Bob', 'Charlie'],
    'Age' :[25 ,30,35] ,
    'Salary': [50000.0, 100000.5, 150000.75],
    'IsMarried': [True, False, True]
})


complection = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        # GPT角色设定
        {"role": "system", "content": "你是一个优秀的数据分析师，现在有这样一份数据集：'%s'" % df_complex},
        # 模拟用户输入信息
        {"role": "user", "content": "请解释一下这个数据集的分布情况"}
    ]
    # 不采用流式输出
    # stream=False
)

print(complection.choices[0].message["content"])
