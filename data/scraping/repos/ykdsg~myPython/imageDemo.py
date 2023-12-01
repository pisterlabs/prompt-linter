import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def demoGenerations():
    # 根据文本指令，向DALL·E 模型发送创建图片的请求，返回图片url
    response = openai.Image.create(
        prompt='a white siamese cat',
        n=1,
        size="256x256",
    )
    print(response["data"][0]["url"])


def demoEdit():
    response = openai.Image.create_edit(
        image=open("sunlit_lounge.png", "rb"),
        mask=open("mask.png", "rb"),
        prompt="A sunlit indoor lounge area with a pool containing a flamingo",
        n=1,
        size="256x256"
    )
    print(response['data'][0]['url'])


def variationsDemo():
    response = openai.Image.create_variation(
        image=open("corgi_and_cat_paw.png", "rb"),
        n=1,
        size="256x256"
    )
    print(response['data'][0]['url'])


if __name__ == '__main__':
    demoGenerations()
    # # demoEdit()
    # variationsDemo()
