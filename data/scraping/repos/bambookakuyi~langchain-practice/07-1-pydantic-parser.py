#!usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

from langchain import OpenAI
model = OpenAI(model_name = "text-davinci-003")

# 创建一个空的DataFrame用于存储结果
import pandas as pd
df = pd.DataFrame(columns = ["flower_type", "price", "description", "reason"])

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
	flower_type: str = Field(description = "鲜花的种类")
	price: int = Field(description = "鲜花价格")
	description :str = Field(description = "鲜花描述")
	reason :str = Field(description = "为什么要写这样的文案")

# 创建输出解析器
from langchain.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object = FlowerDescription)
format_instructions = output_parser.get_format_instructions()
print("输出格式：", format_instructions)
# The output should be formatted as a JSON instance that conforms to the JSON schema below.

# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

# Here is the output schema:
# ```
# {"properties": {"flower_type": {"description": "\u9c9c\u82b1\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u9c9c\u82b1\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u9c9c\u82b1\u63cf\u8ff0", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u5199\u8fd9\u6837\u7684\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}
# ```

# 创建提示模板
from langchain import PromptTemplate
prompt_template = """
您是一位专业的鲜花店文案撰写元。
对于售价为 {price} 元的 {flower}, 您能提供一个吸引人的简短中文描述吗？
{format_instructions}
"""
prompt = PromptTemplate.from_template(prompt_template,
	partial_variables = { "format_instructions": format_instructions })
print("添加partial_varialbles之后的提示：", prompt)
# input_variables=['flower', 'price'] partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.

# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

# Here is the output schema:\n```\n{"properties": {"flower_type": {"description": "\\u9c9c\\u82b1\\u7684\\u79cd\\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\\u9c9c\\u82b1\\u4ef7\\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\\u9c9c\\u82b1\\u63cf\\u8ff0", "title": "Description", "type": "string"}, "reason": {"description": "\\u4e3a\\u4ec0\\u4e48\\u8981\\u5199\\u8fd9\\u6837\\u7684\\u6587\\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}\n```'} template='\n您是一位专业的鲜花店文案撰写元。\n对于售价为 {price} 元的 {flower}, 您能提供一个吸引人的简短中文描述吗？\n{format_instructions}\n'

for flower, price in zip(flowers, prices):
	input = prompt.format(flower = flower, price = price)
	print("输入提示：", input)
	# 您是一位专业的鲜花店文案撰写元。
	# 对于售价为 50 元的 玫瑰, 您能提供一个吸引人的简短中文描述吗？
	# The output should be formatted as a JSON instance that conforms to the JSON schema below.

	# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
	# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

	# Here is the output schema:
	# ```
	# {"properties": {"flower_type": {"description": "\u9c9c\u82b1\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u9c9c\u82b1\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u9c9c\u82b1\u63cf\u8ff0", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u5199\u8fd9\u6837\u7684\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}
	output = model(input)
	parsed_output = output_parser.parse(output)
	parsed_output_dict = parsed_output.dict() # 将Pydantic格式转为字典
	df.loc[len(df)] = parsed_output.dict()

print("输出的字典：", df.to_dict(orient = "records"))
# [{'flower_type': '玫瑰', 'price': 50, 'description': '50元的玫瑰，给你一份甜蜜的表白', 'reason': '因为玫瑰象征着爱情，以50元的价格买一支玫瑰，可以给你一份甜蜜的表白'}, {'flower_type': '百合', 'price': 30, 'description': '一束可爱的百合，清新淡雅，为您的生活增添温馨色彩！', 'reason': '提供吸引人的简短中文描述'}, {'flower_type': 'Carnation', 'price': 20, 'description': '这款康乃馨，鲜艳绚丽，总能打动人心，是节日礼物中的绝佳选择', 'reason': '吸引顾客购买'}]
















