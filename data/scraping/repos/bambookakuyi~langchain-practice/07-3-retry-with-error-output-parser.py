#!usr/bin/env python3

# 1. 构建一个解析时报错的例子
# template变量没用到
# template = """
# Base on ther user question, provide an Action and Action Input for what step should be taken.
# {format_instructions}
# Question: {query}
# Response:
# """

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
class Action(BaseModel):
	action: str = Field(description = "action to taken")
	action_input: str = Field(description = "input to the action")
parser = PydanticOutputParser(pydantic_object = Action)
bad_response = '"action": "search"' # 确实action_inpurt字段
# 直接解析，会报错：langchain.schema.output_parser.OutputParserException: Failed to parse Action from completion "action": "search". Got: Expecting value: line 1 column 1 (char 0)
# parser.parse(bad_response)

# 2. 使用RetryWithErrorOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
  template = "Answer the user query.\n{format_instructions}\n{query}\n",
  input_variables = ["query"],
  partial_variables = { "format_instructions": parser.get_format_instructions() }
)
prompt_value = prompt.format_prompt(query = "what are the colors of Orchid?")

from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI
retry_parser = RetryWithErrorOutputParser.from_llm(
	parser = parser,
	llm = OpenAI(temperature = 0)
)
parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print("重试后结果：", parse_result)
# 重试后结果： action='search' action_input='What are the colors of Orchid?'


# 要点：
# OutputFixingParser 只能做简单的格式修复，如果出错的不仅是格式，比如输出不完整，有缺失内容，那么就可以用RetryWithErrorOutputParser来重试
