#!usr/bin/env python3

# 1. 先设计一个解析时出现错误的例子
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Flower(BaseModel):
	name: str = Field(description = "name of flower")
	color: List[str] = Field(description = "the colors of this flower")
flower_query = "Generate the charaters for a random flower."
# 假设Model返回结果如下，正确格式应该为：'{"name": "康乃馨", "colors": ["粉红色","白色","红色"]}'
misformatted = "{'name': '康乃馨', 'colors': ['粉红色', '白色', '红色']}"
# 创建一个用于解析输出的Pydantic解析器，此处希望解析为Flower格式
parser = PydanticOutputParser(pydantic_object = Flower)
# 报错：langchain.schema.output_parser.OutputParserException: Failed to parse Flower from completion {'name': '康乃馨', 'colors': ['粉红色', '白色', '红色']}. Got: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
# parser.parse(misformatted)

# 2. 使用 OutputFixingParser 来自动结果类似的格式错误
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from dotenv import load_dotenv
load_dotenv()

new_parser = OutputFixingParser.from_llm(parser = parser, llm = ChatOpenAI())
# 如果解析失败，会将格式错误的输出以及格式化的指令传递给大模型，并要求LLM进行相关修复
result = new_parser.parse(misformatted)
print("修正格式后的结果：", result)
# 修正格式后的结果： name='康乃馨' color=['粉红色', '白色', '红色']