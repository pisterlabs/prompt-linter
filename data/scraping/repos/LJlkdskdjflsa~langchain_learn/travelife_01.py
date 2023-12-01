from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain import PromptTemplate

# 初始化 langchain
llm = OpenAI()


template = """你是一個有經驗的旅行者，請根據以下需求，給我三個 50 字的旅遊行程簡介（使用繁體中文）
需求：{requirement}
請幫我把行程簡介寫成為一個列表，每個行程簡介之間用「/」隔開
例子：
日本賞楓/日本賞櫻/日本賞雪
"""

pt = PromptTemplate(
    input_variables=["requirement"],
    template=template,
)

pt.format(requirement="我想去日本玩，但是不知道要怎麼玩")


llm.predict(pt.format(requirement="我想去日本玩，但是不知道要怎麼玩"))


def format_itinerary_list(itinerary_str):
    # 使用 '/' 分隔字符串
    items = itinerary_str.split("/")

    # 移除每個項目前後的空白字符
    items = [item.strip() for item in items if item.strip()]

    return items


# 使用範例
itinerary_str = "\n1. 日本之旅：體驗日本的精彩文化，探索古堡、寺院、茶室等歷史文化遺產，還可以品嘗地道的日式料理。/ \n2. 登山旅行：登上日本的名山，欣賞壯觀的自然景觀，擁抱大自然的美景。/ \n3. 渡假度假：在日本的溫泉區，放鬆身心，在溫泉中享受美好時光。"
formatted_list = format_itinerary_list(itinerary_str)
print(formatted_list)
