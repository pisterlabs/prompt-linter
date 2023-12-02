import pandas as pd
import openai
import json
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

global ouput_path 
output_path = config.OUTPUT_PATH

openai.api_key = '' # 填入你的opai api key
data = pd.read_csv(ouput_path, header=None, encoding='unicode_escape', dtype={0: str, 4: str, 3: str})
with open('./ava.json', 'r', encoding='utf-8') as json_file:
    ava_data = json.load(json_file)
def json_to_text(actions):
    text = []
    for action in actions:
        json_code = action
        action_text = ava_data.get(str(json_code))
        text.append(action_text)
    return text

def replace_labels(row):
    actions = [int(action[1]) for action in re.findall(r"\((?:tensor\(.+?\)),\s*(\d+)\)", row[3])]
    actions_text = json_to_text(actions)
    return actions_text

# 替换'action1'列的标签为文本
data[3] = data.apply(replace_labels, axis=1)

# 定义NLP模型生成句子的函数
def generate_sentence(scene, name, actions, position):
    action_str = ' '.join(actions)
    prompt = f"现在你将作为一个将文字拼接起来并讲述一段完整视频画面的机器人，图像识别并导出数据使用的是slowfast，视频是标准1080p格式，你将获得如下参数：label也就是场景数表示该对象使哪个场景；position，position是一个有四个元素的数组，从第一二个元素组成的数组表示物体检测框左上角在视频该帧中的坐标，第三四个元素组成的数组表示该目标检测框的长宽数值；name：用于获取是哪个人，并在之后的片段中你要记得这是同一个人并避免太过离谱的故事讲述，最后你将获取到一个action的参数，用于告诉你这个框里名叫name的人正在做什么，你需要综合不同的action编出合理的故事。故事的要求如下，能描述出谁在干什么，并更关注框更大的人，因为她/他是该场景的主角，多判断人物与人物，人物与场景之间的交互。你只需要讲故事，不需要做任何分析。你明白了吗？同时重复的内容请你不要反复输出。因为这是对场景的描述，你懂吗？请你根据我提供的信息,用中文描述这个场景，人名也翻译为中文，比如jack翻译为杰克。\n场景：{scene}。\n称呼：{name}。\n位置信息：{position}。\n描述：{action_str}"

    # 创建新的Completion对象来开始一个新的会话
    completion = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1000,  # 设置生成句子的最大长度
        n=1,  # 设置生成句子的数量
        stop=None,  # 设置停止条件
        temperature=0.5,  # 设置生成句子的多样性，较高的温度会生成更多变化的句子
        top_p=0.4,  # 设置生成句子的概率阈值，较小的值会生成更保守的句子
        frequency_penalty=0,  # 设置频率惩罚，较大的值会降低重复性
        presence_penalty=0,  # 设置存在惩罚，较大的值会鼓励更全面的句子
    )

    generated_sentence = completion.choices[0].text.strip()  # 获取生成的句子

    return generated_sentence

# 生成句子
data['句子'] = data.apply(lambda row: generate_sentence(row[0], row[4], row[3], row[2]), axis=1)

# 将生成的句子保存到txt文件
data['句子'].to_csv('generated_sentences.txt', index=False, header=False, encoding='utf-8-sig')

# 只保留生成的句子列并保存为新的CSV文件
data = data[['句子']]
data.to_csv(output_path +'generated_data.csv', index=False, encoding='utf-8-sig')