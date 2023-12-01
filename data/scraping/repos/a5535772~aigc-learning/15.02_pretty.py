import myconfig
import openai, os
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains import LLMRequestsChain
from langchain.chains import TransformChain, SequentialChain
from tool_15_02 import parse_weather_info

# open ai api key
openai.api_key = os.environ.get("OPENAI_API_KEY")


# build LLMRequestsChain
def build_requests_chain() -> LLMRequestsChain:
    # 定义template
    template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
    请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
    请使用如下格式：
    Extracted:<answer or "找不到">
    >>> {requests_result} <<<
    Extracted:"""

    prompt_template = PromptTemplate(
        input_variables=["query", "requests_result"],
        template=template,
    )

    return LLMRequestsChain(llm_chain=LLMChain(llm=OpenAI(temperature=0), prompt=prompt_template))


"""
transform_func函数接收一个字典类型的输入'inputs'，其中应该包含键"output"
提取与键"output"相关联的值，并赋给变量'text'
调用parse_weather_info函数，使用'text'作为参数解析天气信息
将解析后的天气信息存储在一个新的字典中，键为"weather_info"
返回转换后的字典作为函数的输出。
"""


def transform_func(inputs: dict) -> dict:
    # 使用键"output"从输入字典中提取文本数据
    text = inputs["output"]
    # 调用parse_weather_info函数从文本中解析天气信息
    weather_info = parse_weather_info(text)
    # 创建一个新的字典，包含解析后的天气信息
    transformed_data = {"weather_info": weather_info}
    # 返回转换后的数据
    return transformed_data


"""业务调用"""

question = "今天上海的天气怎么样？"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}

requests_chain = build_requests_chain()
transformation_chain = TransformChain(input_variables=["output"], output_variables=["weather_info"],
                                      transform=transform_func)
sequential_chain = SequentialChain(chains=[requests_chain, transformation_chain],
                                   input_variables=["query", "url"], output_variables=["weather_info"])
final_result = sequential_chain.run(inputs)
print(final_result)
