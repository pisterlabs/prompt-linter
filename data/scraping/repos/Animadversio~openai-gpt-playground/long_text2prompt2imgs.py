
import os
import openai
import time
import pickle as pkl

import json
from pathlib import Path
rootdir = r"E:\DL_Projects\NLP\Poem2Art"
openai.api_key = os.getenv("OPENAI_API_KEY")

text = """
我看到了我的爱恋
我飞到她的身边
我捧出给她的礼物
那是一小块凝固的时间
时间上有美丽的条纹
摸起来像浅海的泥一样柔软
她把时间涂满全身
然后拉起我飞向存在的边缘
这是灵态的飞行 （灵态的飞行）
我们眼中的星星 （像幽灵）
星星眼中的我们也像幽灵
时空全都是锋利的羽翼
多维宇宙任我自由地横行
跌落二维里化作了泡影
只有串串空荡荡的回音
几个纪元前静默的生命
手捧火球变得如此的坚定
谁能守住时间宏大的秘密
如此笃定 便给岁月以文明
我看到了我的爱恋
我飞到她的身边
我捧出给她的礼物
那是一小块凝固的时间
时间上有美丽的条纹
摸起来像浅海的泥一样柔软
她把时间涂满全身
然后拉起我飞向存在的边缘
这是灵态的飞行 （灵态的飞行）
我们眼中的星星 （像幽灵）
星星眼中的我们也像幽灵
时空全都是锋利的羽翼
多维宇宙任我自由地横行
跌落二维里化作了泡影
只有串串空荡荡的回音
几个纪元前静默的生命
手捧火球变得如此的坚定
谁能守住时间宏大的秘密
如此笃定 便给岁月以文明
"""
usage_counter = {"completion_tokens": 0,  "prompt_tokens": 0,  "total_tokens": 0}
#%%
lines = text.strip().split("\n")
bsize = 1
# text_description = "海洋在我们头顶，天空在脚下，头顶的海洋向我们砸了过来像一场暴雨，劈头盖脸浇在船顶上"

text_translation_pairs = []
for i in range(0, len(lines), bsize):
    line_batch = lines[i:i + bsize]
    text_description = " ".join(line_batch)
    query = f"""
    "{text_description}" translate this into an English prompt for MidJourney. Just output the prompt between " " without any text before or after it
    """

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=250,
                messages=[
                {"role": "system", "content":
                 "You are a helpful and knowledgeable art assistant, "
                      "helping people to translate chinese text in to English description for generative art."},
                {"role": "user", "content": query},
                ]
            )
            break
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting 5 seconds and trying again...")
            time.sleep(1)
        continue
    # print(completion)
    answer = completion["choices"][0]["message"]["content"]
    print(text_description, "\n", answer)
    text_translation_pairs.append((text_description, answer))
    for k, v in completion["usage"].items():
        usage_counter[k] += v
print(usage_counter)

#%%
with open(Path(rootdir)/"text_translation_pairs_singleline.pkl", "wb") as f:
    pkl.dump(text_translation_pairs, f)
# json
with open(Path(rootdir)/"text_translation_pairs_singleline.json", "w", encoding="utf-8") as f:
    json.dump(text_translation_pairs, f, separators=(',', ':'), ensure_ascii=False, indent=2)
#%%
system_message = """
You are going to pretend to be Concept2PromptAI or C2P_AI for short. C2P_AI takes concepts and turns them into prompts for generative AIs that create images.

You take a concept or sentense then provide a prompt for it as a text string in " ".

Use the following examples as a guide:

Concept: A macro shot of a stempunk insect

Prompt: a close up of a bug with big eyes, by Andrei Kolkoutine, zbrush central contest winner, afrofuturism, highly detailed textured 8k, reptile face, cyber steampunk 8 k 3 d, c 4 d ”, high detail illustration, detailed 2d illustration, space insect android, with very highly detailed face, super detailed picture 

Concept: An orange pie on a wooden table

Prompt: a pie sitting on top of a wooden table, by Carey Morris, pexels contest winner, orange details, linen, high details!, gif, leafs, a pair of ribbed, vivid attention to detail, navy, piping, warm sunshine, soft and intricate, lights on, crisp smooth lines, religious 

Concept: a close up shot of a plant with blue and golden leaves

Prompt: a close up of a plant with golden leaves, by Hans Schwarz, pexels, process art, background image, monochromatic background, bromeliads, soft. high quality, abstract design. blue, flax, aluminium, walking down, solid colours material, background artwork 
"""

# text_translation_pairs = pkl.load(open(Path(rootdir)/"text_translation_pairs.pkl", "rb"))
text_translation_pairs = pkl.load(open(Path(rootdir)/"text_translation_pairs_singleline.pkl", "rb"))
text_prompt_pairs = []
for i, (text_cn, text_en) in enumerate(text_translation_pairs):
# for text_cn, text_en in [("跌落二维里化作了泡影", "falling into two-dimensional space and becoming a bubble. Use this phrase as inspiration to create a generative artwork that explores themes of transformation, ephemerality, and the boundaries between different states of being."),
#                          ("我捧出给她的礼物", "I present a gift to her.")]:
# text_cn, text_en = "前进，前进，不择手段的前进！", "Move forward, forward, forward at all costs!"
# for text_cn, text_en in [("孩子们，家已经变成一副画了！", "Children, our home has become a painting!"),
#                     ("宇宙很大, 生活更大, 我們一定還能相見的", "The universe is big, life is bigger, we will meet again"),
#                     ("我們度過了幸福的一生", "We have spent a happy life together"),
#                     ("歡迎你們來到647號宇宙", "Welcome to the Universe 647"),]:
    query = f"""
    The concept to be visualized is "{text_en}" translate this into an English prompt for MidJourney.
    """
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=250,
                messages=[
                    {"role": "system", "content":
                    "You are a helpful and knowledgeable art assistant, "
                         "helping people to translate chinese text in to English description for generative art." },
                    {"role": "user", "content":system_message +"\n"+ query},
                ]
            )
            break
        except openai.error.APIConnectionError:
            print("API connection error. Waiting 5 seconds and trying again...")
            time.sleep(1)
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting 5 seconds and trying again...")
            time.sleep(1)
        except openai.error.APIError:
            print("API error. Waiting 5 seconds and trying again...")
            time.sleep(1)
        continue
    # print(completion)
    answer = completion["choices"][0]["message"]["content"]
    print(text_cn, "\n", text_en, "\n", answer)
    text_prompt_pairs.append((text_cn, text_en, answer))
    for k, v in completion["usage"].items():
        usage_counter[k] += v
    print(usage_counter)
#%%
def price_func(usage_counter):
    return usage_counter['total_tokens'] * 0.002 / 1000


print("Total cost $", price_func(usage_counter))

#%%
with open(Path(rootdir)/"text_prompt_pairs_singleline.pkl", "wb") as f:
    pkl.dump(text_prompt_pairs, f)
# json
with open(Path(rootdir)/"text_prompt_pairs_singleline.json", "w", encoding="utf-8") as f:
    json.dump(text_prompt_pairs, f, separators=(',', ':'), ensure_ascii=False, indent=2)
#%%
# text_prompt_pairs = json.load(open(Path(rootdir)/"text_prompt_pairs_singleline.json", "r", encoding="utf-8"))
#%%
with open(Path(rootdir)/"text_prompt_pairs.pkl", "wb") as f:
    pkl.dump(text_prompt_pairs, f)
# json
with open(Path(rootdir)/"text_prompt_pairs.json", "w", encoding="utf-8") as f:
    json.dump(text_prompt_pairs, f, separators=(',', ':'), ensure_ascii=False, indent=2)


