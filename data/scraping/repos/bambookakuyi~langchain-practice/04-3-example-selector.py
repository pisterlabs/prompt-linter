#!/usr/bin/env python3

# 要点：
# 如果示例太多，一次性发送给模型是比较抵消的，且容易消耗Token跟流量。
# LangChain提供了实例选择器，通过向量相似度比较的功能，来选择最合适的样本。

# 1. 创建一些示例
samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]

# 2. 创建提示模板
from langchain.prompts.prompt import PromptTemplate
template = "鲜花类型：{flower_type}\n场合：{occasion}\n文案：{ad_copy}"
prompt_sample = PromptTemplate(
	input_variables = ["flower_type", "occasion", "ad_copy"],
	template = template)


# 3. 使用示例选择器
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma # 向量数据库 pip3 install chromadb
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
example_selector = SemanticSimilarityExampleSelector.from_examples(
  samples,
  OpenAIEmbeddings(),
  Chroma,
  k = 1
)

# 4. 创建一个使用示例选择器的FewShotPromptTemplate对象
from langchain.prompts.few_shot import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
	example_selector = example_selector,
	example_prompt = prompt_sample,
	suffix = "鲜花类型：{flower_type}\n场合：{occasion}",
	input_variables = ["flower_type", "occasion"]
)
input = prompt.format(flower_type = "野玫瑰", occasion = "爱情")
print(input)

# 4. 把提示传递给大模型
from langchain.llms import OpenAI
model = OpenAI(model_name = "text-davinci-003")
result = model(input)
print(result)
# 结果：文案：野玫瑰，凌厉的外表，里面洋溢着温柔的情愫，传达你无尽的爱意。



