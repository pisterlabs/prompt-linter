import json
from time import sleep
from typing import Optional
from openai import OpenAI
from httpx import Client
import nlp_pb2_grpc
from nlp_pb2_grpc import NlpServicer
from nlp_pb2 import (
    NerRequest,
    NerReply,
)


class NlpServer(NlpServicer):
    def Ner(self, request: NerRequest, context):
        ner_ret = ner(request.text, request.field)
        return NerReply(ner_ret=ner_ret)


from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    # In docker, do not need to set proxy, for it uses host network which does.
    # http_client=Client(proxies="http://127.0.0.1:7890"), timeout=30, max_retries=0
)


def ner(text: str, field: str) -> str:
    s = f"You are an assistant capable of performing named entity recognition and \
proficient in knowledge related to {field}. I need you to extract the named entities \
from the following content and return each named entity in JSON format along with **more \
detailed relevant information**. Remember to use the Markdown syntax with \"```\" to enclose the JSON, \
besides, use entity name as key and attributes as value, and in attributes, use attribute name as key. \
Remember to use your knowledge about {field} to enrich the attributes."
    q = f"Following text is about {field}. Please Perform named entity recognition on the following \
content and return each named entity in JSON format along **with more detailed relevant information**: \n{text}"
    print(f"Asking GPT: \n{q}")
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": s,
                    },
                    {
                        "role": "user",
                        "content": q,
                    },
                ],
            )
            content = completion.choices[0].message.content
            print(f"GPT answer:\n {content}\n")
            ret = extract(content if content else "")
            try:
                json.loads(ret)
            except Exception as e:
                print(e)
                continue
        except Exception as e:
            print(e)
            sleep(20)
            continue
        break
    return ret


def extract(content: str) -> str:
    flag = False
    ret = []
    for line in content.split("\n"):
        if line.startswith("```"):
            flag = not flag
            continue
        if flag:
            ret.append(line)
    return "\n".join(ret)


if __name__ == "__main__":
    print("Start test ...")
    ret = ner('''域名解析
cqu.edu.cn 其中 cqu 是主机名；edu 为机构性域名；cn 是地理域名
大数据
指无法在一定时间范围内用常规软件工具进行捕捉、管理和处理的数据集合，需要新的处理模式才能具有更强的决策力、洞察力和流程优化能力来适应海量、高增长率和多样性的信息资产
特征
海量化：体量大
多样化：类型多样（视频、图片、文字等），无明显模式，不连贯的语义或句义
快速化：实时分析而非批量分析，处理快，增长快
价值化：低价值密度，高商业价值
思维方式
全样而非抽样
效率而非精确
相关而非因果
大数据技术起源
Google 的 GFS (Google file system)、MapReduce (并行计算编程)、BigTable (Google 的分布式数据储存系统) 但是不开源（用的cpp） 开源 Hadoop 系统（用的 java ，如今有大量其他语言的实现）： HDFS Hadoop MapReduce HBase
云计算
IaaS 基础设施即服务
PaaS 平台即服务
SaaS 软件即服务
FaaS 函数即服务
aa 即 as a
''', "物联网")
    print(ret)
