import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    SystemMessage
)
import os
import re


def get_vale(history: list):

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    chat = ChatOpenAI(temperature=0)

    prompt = """
    从现在开始，你是一个语义分析bot，我会给你一段两个人之间三段聊天的内容。你需要总结这三段聊天的内容来分析糖糖的情感变化。
    并且输出聊天数值的变化在该表格的小括号内：[好感度]:() [压力]:() [阴暗度]:() 
    具体要求如下：
    【人物限制】：你只可以分析在这一段对话中糖糖这一个角色的情感变化。
    【数值要求】：你进行分析之后，需要严格按照一个表格进行输出，该表格的格式要求如下：[阴暗值]:() [压力]:() [阴暗度]:()。
    【回复要求】：你只可以输出添加数值后的表格。并且数值的变化量只可以在0至10之间，且需要填写在小括号内！
    """
    test = [('阿p:「午安午安~」', '糖糖:「午安午安~阿P，睡得好吗？有做个美梦吗？哈哈哈」'),
            ('阿p:「午安午安~」', '糖糖:「真是懒猪呀，又睡了一整天。不过没关系，反正我们在一起就好啦~」'),
            ('阿p:「睡觉啦~」', '糖糖:「好好好，安安睡个好觉，做个美梦哦~晚安晚安，我会一直陪着你的！」')]

    # 获取到列表的最后三个值
    strs = str(history[-3:])

    # 使用langchain让chatgpt根据strs的内容,来获得指定输出。
    messages = [
        SystemMessage(content=prompt),
        AIMessage(content="好的，请为我提供对话聊天内容，我会总结里面糖糖的角色感情变化，并且输出指定格式的内容给您"),
        SystemMessage(content=str(test)),
        AIMessage(content='[压力]:(0) [好感度]:(+4)  [阴暗度]:(0)'),
        SystemMessage(content=str(strs)),
    ]
    result = chat(messages)

    return result


def re_(strs):
    # 处理字符串，并且返回指定的值
    # 例如，传入的str为content='[压力]:(+2) [好感度]:(-2)  [阴暗度]:(+1)' 则返回列表：[-2,+2,+1]
    result = []  # 创建一个空列表，用于存放结果
    pattern = r'\(([-+]?\d+)\)'  # 创建一个正则表达式模式，用于匹配括号内的数值
    re_str = str(strs)
    matches = re.findall(pattern, re_str)  # 使用findall方法，找出字符串中所有符合模式的子串
    for match in matches:  # 遍历所有的子串
        num = int(match)  # 将子串转换为整数
        result.append(num)  # 将整数添加到结果列表中
    return result  # 返回结果列表


if __name__ == '__main__':
    # print(get_vale(a))
    # 测试函数
    a = [
        ('阿p:「睡觉啦~」', '糖糖:「好好好，安安睡个好觉，做个美梦哦~晚安晚安，我会一直陪着你的！」'),
        ('阿p:「在干嘛？」', '糖糖:「嘿嘿，我在看一部新番呢，剧情超级精彩的！你呢，阿P在干嘛捏？」'),
        ('阿p:「在干嘛？」', '糖糖:「嘿嘿，就在家里无所事事呀，你呢阿P？」'),
        ('阿p:「准备上课了~」', '糖糖:「诶～上什么学啊来当主播啊！！！！！」↑发了这句过去。」'),
        ('阿p:「好！明天开始就当vup」', '糖糖:「阿哈，阿P你太厉害了！以后我们一起变成传说的VUP吧！加油加油！」'),
        ('阿p:「```吃饱啦，想睡午觉了~```」', '糖糖:「啊～午睡真是人生一大享受啊！阿P，一起来做个美梦吧，我也好想睡觉呀～」'),
        ('阿p:「你好烦！」', '糖糖:「哈哈，阿P你只是开玩笑阿吧」'),
        ('阿p:「你觉得呢?」', '糖糖:「阿p，不要这样，我有些害怕」'),
        ('阿p:「呵呵」', '糖糖:「阿p！你不要这样好不好?求你了」')]

    content = "content='[好感度]:(-2) [压力]:(+2) [阴暗度]:(+1)'"
    print(re_(content))
