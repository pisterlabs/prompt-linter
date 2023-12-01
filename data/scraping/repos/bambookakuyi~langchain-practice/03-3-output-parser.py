#!usr/bin/env python3

from langchain import PromptTemplate, OpenAI
from dotenv import load_dotenv
load_dotenv()

prompt_template = """
您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 的 {flower_name}，您能提供一个吸引人的简短描述吗？
{format_instructions}
"""

model = OpenAI(model_name = "text-davinci-003")

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# 定义我们要接收的相应格式
response_schemas = [
  ResponseSchema(name = "description", description = "鲜花的描述文章"),
  ResponseSchema(name = "reason", description = "为什么要写这样的文章")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 获取格式指示
format_instructions = output_parser.get_format_instructions()
# 根据原始模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template,
	partial_variables = { "format_instructions": format_instructions })

flowers = ["百合", "康乃馨"]
prices = ["30", "20"]

# 创建一个空的DataFrame用于存储结果
import pandas as pd # pip3 install pandas
# 声明列名
df = pd.DataFrame(columns = ["flower", "price", "description", "reason"])
for flower, price in zip(flowers, prices):
	input = prompt.format(flower_name = flower, price = price)
	output = model(input)

	# 解析模型输出（这是一个字典结构）
	parsed_output = output_parser.parse(output)
	# 在解析后的输入中添加"flower"和"price"
	parsed_output["flower"] = flower
	parsed_output["price"] = price

	# 将解析后的输添加到DataFrame中
	df.loc[len(df)] = parsed_output

print(df.to_dict(orient = "records"))
# 结果
# [
#   {'flower': '百合', 'price': '30', 'description': '30 的百合是一种美丽而优雅的鲜花，它的白色苞片代表着纯洁和爱的美好，是表达爱意的最佳选择。', 'reason': '帮助客户理解此价位的鲜花的美丽和优雅，激发他们购买此价位的百合鲜花的热情。'},
#   {'flower': '康乃馨', 'price': '20', 'description': 'Treat yourself to a classic symbol of beauty and love with a bouquet of 20 carnations. Each flower is carefully grown and arranged to create a stunning, vibrant display that will add a touch of elegance to any occasion.', 'reason': 'The description of the carnations is intended to evoke feelings of beauty and love, as well as emphasize the quality of the flowers in order to encourage customers to purchase them.'}
# ]


# 要点：
# 模型只能生成文案，通过解析模型输出的功能可以将结果解析成结构化数据


