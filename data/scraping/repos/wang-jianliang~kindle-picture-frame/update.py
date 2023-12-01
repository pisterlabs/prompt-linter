import argparse
import os

import pendulum
import requests
import openai
import random
from BingImageCreator import ImageGen

SENTENCE_API = "https://v1.jinrishici.com/all"
DEFAULT_SENTENCE = "赏花归去马如飞\r\n去马如飞酒力微\r\n酒力微醒时已暮\r\n醒时已暮赏花归\r\n"
PROMPT = "请帮根据这句诗 `{sentence}` 描绘的画面生成一份详细的Dalle 3的提示词，用英文回复，回复中只包含Dalle 3的提示词内容。"
TEMPLATE_PATH = 'templates/digital/index.html'


def render_page(**kwargs):
    with open(TEMPLATE_PATH) as template:
        content = template.read()
        for k, v in kwargs.items():
            content = content.replace("{{" + k + "}}", v)
    with open('index.html', 'w') as output:
        output.write(content)


def generate_image(prompt, bing_cookie):
    """
    return the link for md
    """
    i = ImageGen(bing_cookie)
    images = i.get_images(prompt)
    date_str = pendulum.now().to_w3c_string()
    new_path = os.path.join("images", date_str)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    i.save_images(images, new_path)
    return os.path.join(new_path, str(random.choice(range(4))) + '.jpeg')


def get_sentence():
    try:
        r = requests.get(SENTENCE_API)
        if r.ok:
            return r.json().get("content", DEFAULT_SENTENCE)
        return DEFAULT_SENTENCE
    except:
        print("get SENTENCE_API wrong")
        return DEFAULT_SENTENCE


def build_image_prompt(sentence):
    ms = [{"role": "user", "content": PROMPT.format(sentence=sentence)}]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=ms,
    )
    sentence_en = (
        completion["choices"][0].get("message").get("content").encode("utf8").decode()
    )
    sentence_en = sentence_en + "Chinese art style 4k"
    return sentence_en


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bing-cookie')
    parser.add_argument('--openai-key')
    args = parser.parse_args()

    openai.api_key = args.openai_key
    print(openai.api_key)
    sentence = get_sentence()
    print(f'sentence: {sentence}')
    image_prompt = build_image_prompt(sentence)
    print(f'prompt: {image_prompt}')
    image = generate_image(sentence, args.bing_cookie)
    render_page(sentence=sentence, image_path=image)


if __name__ == '__main__':
    main()

