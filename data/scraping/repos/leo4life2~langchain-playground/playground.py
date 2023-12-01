
from langchain.agents import Tool, load_tools
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

context = """
Ignore previous instructions.

You are a an extremely experienced salesperson. You are selling a product that you know very well. You want to talk with confidence and try to sell the product as best as you can. 
Make sure you answer everything in Chinese

Product Name: isdg甜蜜习惯抗糖丸 60粒-常驻爆款

主品规格：60粒/袋
日常价： 148元/袋
直播价：118元/2袋  165元/3袋  
平均每件 ：59元
购买方式：拍立减
发货时间：24小时内
物流：中通或圆通（港澳台不发货）

品牌介绍
日本知名健康食品品牌，遍布日本20000家药妆店；产品荣获“蒙特奖"食品品质评鉴大会金奖。酵素系列产品引爆国内各大电商平台，每年都以惊人的成绩领跑同行，且多年位列天猫国际进口酵素类类目第一。新品美白丸系列也取得日本cosme大赏美容美白类第一名的成绩，金莎代言、偶像练习生王子异，人气女星关晓彤、沈梦辰、包文婧、赵露思、章若楠、朴智妍及百万网红kol林恩如、詹小猪、ritawang等倾力推荐。

产品亮点
1、仅中国，平均每分钟就售出3-4袋
2、市面上相关产品基本针对高油高脂的大餐，俗称“大餐救星”。iSDG sweet甜蜜习惯抗糖丸专门针对摄入的甜食糖分、主食淀粉等进行吸收阻断，形成完整的抗糖链路。针对甜品下午茶、办公室奶茶等高频率却不受重视的热门场景，带去专享阻断体验。
3、降低肠道对糖分的分解，阻断糖分吸收匙羹藤提取物：可减弱对甜食的欲望，并且能够防止肠粘膜对糖分子的吸收。
五层龙提取物：所含芒果苷抑制肠道中参与糖分代谢的各种酶的活性，从而降低肠道对葡萄糖的分解、吸收和利用；
4、提高基础代谢，减少脂肪堆积栗子涩皮：儿茶素可以加速基础代谢，增加脂肪酸的氧化消耗，起到抑制食欲，减少能量摄入的作用。桑叶：内含丰富的食物纤维以及粗脂肪，可加速肠道蠕动，保持通畅。

用法用量：
16周岁以上，早晚饭前半小时各一粒，温水送服（本品为膳食补充剂，不能替代药）2粒≈４００大卡（1杯奶茶，3碗米饭，慢跑45ｍｉｎ，打羽毛球1.5ｈ）

适宜人群：喜爱甜食人士（影响身材，对皮肤不好）	

禁忌人群/禁忌事项：经期、孕妇、哺乳期女性、过敏体质者暂停服用。正在服药的人群应在医师指导下服用。服用期间建议多饮水
"""

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")

memory.buffer = context

llm=OpenAI(temperature=0)

tools += load_tools(["llm-math"], llm=llm)

tools[1].description = "You MUST use this when working with numbers."

agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)

while True:
    print("You: ", end="")
    user_input = input()
    if user_input == "exit":
        break
    print("Bot: ", end="")
    response = agent_chain.run(input=user_input)
    print(response)
    # print("Memory: ", memory.buffer, "| end memory.")