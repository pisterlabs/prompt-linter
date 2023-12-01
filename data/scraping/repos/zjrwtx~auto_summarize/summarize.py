#基于python和openai的api为文章自动生成摘要，只需要提供原文链接
#公众号【正经人王同学】
#日期2022-12-6

#导入相关库
import requests
import json
import openai

#设置自己的api密钥（到openai官网获取）
openai.api_key = "example"


#定义生成文章摘要的函数
def summarize_essay(essay_url):

    #从给的url里读取文章
    essay_text=requests.get(essay_url).text

    #用openai生成文章摘要
    summary_response=openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Summarize the following text:\n{essay_text}",
        max_tokens=500,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    #从API响应中提取摘要文本
    summary_text=summary_response["choices"][0]["text"]

    #将摘要文本拆分为一个项目要点列表并返回结果
    bullet_points=summary_text.split("10")
    return bullet_points

#测试摘要生成函数
bullet_points=summarize_essay("http://www.paulgraham.com/read.html")

#显示最后的摘要结果
for bullet_point in bullet_points:
    print(bullet_point)
