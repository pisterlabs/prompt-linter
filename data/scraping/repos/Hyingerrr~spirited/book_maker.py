import base64
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import requests
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from prompts import *


class BookMaker:
    def __init__(self, prompts, model_name, styles, chapterNum=3):
        # 描述词
        self.prompts = prompts
        # 模型名称
        self.llm = ChatOpenAI(model=model_name)
        # 风格类型
        self.story_styles = styles
        # 章节数
        self.chapterNum = chapterNum
        #  标题
        self.story_title = ''
        # 描述
        self.sd_pages_prompts = []
        # 图片地址
        self.images_urls = []

        self.progress_locked = threading.Lock()
        self.progress = st.progress(10, "初始化...")
        # 项目路径
        self.abspath = os.path.dirname(os.path.abspath(__file__))
        # self.runner()

    def runner(self) -> list[tuple]:
        self.set_progress(0.1, '正在创作故事...')
        # 生成故事描述（含标题）
        story_pages = self.generate_story()

        # 生成SD的Prompt
        self.set_progress(0.25, "正在生成Prompts...")
        self.make_pages_prompt(story_pages)

        # 生成绘本图片页 pageList
        self.set_progress(0.4, '正在创建图像...')
        self.images_urls = self.text_to_images()
        if len(self.images_urls) == 0:
            raise ValueError('images_urls is not empty.')

        print("图片地址：")
        print(self.images_urls)

        self.set_progress(0.7, '正在转换PDF...')
        # 合成一个元组
        return list(zip(self.images_urls, story_pages))

    def set_progress(self, current: float, description):
        """
        设置进度
        :param current:  当前进度
        :type current: float
        :param description: 进度描述
        :type description: basestring
        :return:
        :rtype:
        """

        # 控制锁
        with self.progress_locked:
            self.progress.progress(current, description)
        return

    def generate_story(self) -> list:
        """
            response like below:
            [
                {
                "title": "",
                "Page 1": "",
                "Page 2": "",
                }
            ]
        """

        # parts = BOOK_DESC_PROMPT.split("{}")
        # desc = parts[0] + str(self.chapterNum) + parts[1]
        story_prompt = BOOK_DESC_PROMPT.format(self.chapterNum) + self.prompts
        print('故事Prompt')
        print(story_prompt)

        story = self.llm([HumanMessage(content=story_prompt)])
        stories = re.split('Page \d+:', story.content)
        print("故事：")
        print(stories)
        self.story_title = stories[0].strip("Title: \n")
        return stories

    def make_pages_prompt(self, story_pages: list):
        story_pages = list(story_pages)
        # 删除第一元素（标题）
        story_pages.pop(0)

        prompt1 = f'Use the current function to generate a visual description and effect that matches the storyline ' \
                  f'of this book ' \
                  f'which should be relevant to the storyline. {story_pages}'
        response = self.llm([HumanMessage(content=prompt1)], functions=get_book_sentiment_atmosphere)
        required = json.loads(response.additional_kwargs.get('function_call').get('arguments'))

        def generate_prompt(passage, require):
            prompt2 = f'Generate book info: {require}. Generate style {self.story_styles} Current Page: {passage}' \
                      f' Generate a visual description of the passage using the function. ' \
                      f' Creatively fill all parameters with guessed/assumed values if they are missing. within 1500 ' \
                      f'characters'
            prompt_res = self.llm([HumanMessage(content=prompt2)], functions=get_visual_description)
            return json.loads(prompt_res.additional_kwargs.get('function_call').get('arguments'))

        # 并行生成故事情节
        with ThreadPoolExecutor(max_workers=1) as executor:
            page_prompt_list = list(executor.map(generate_prompt, story_pages, [required] * len(story_pages)))

        # 需要组装sd的页prompt
        for _, page in enumerate(page_prompt_list):
            sd_prompt = ""
            if page['base_setting']:
                sd_prompt += f"{page['base_setting']}, "
            if page['setting']:
                sd_prompt += f"{page['setting']}, "
            if page['time_of_day']:
                sd_prompt += f"{page['time_of_day']}, "
            if page['weather']:
                sd_prompt += f"{page['weather']}, "
            if page['key_elements']:
                sd_prompt += f"{page['key_elements']}, "
            if page['specific_details']:
                sd_prompt += f"{page['specific_details']}  "

            sd_prompt += f"{required['lighting']}, {required['mood']}, {required['color_palette']}," \
                         f" {required['page_summary']} in the style of {self.story_styles}"
            self.sd_pages_prompts.append(sd_prompt)

        if len(story_pages) != len(self.sd_pages_prompts):
            raise ValueError('prompts与绘本页面不匹配, 请重试。')

        print("故事SD页面prompts：")
        print(self.sd_pages_prompts)

    def text_to_images(self) -> list:
        print("正在创建图片，请稍后...")

        def generate_image(i, prompt):
            print(f'第{i + 1}页：{prompt}')

            # engine_id = 'stable-diffusion-xl-1024-v1-0'
            engine_id = os.getenv('STABILITY_SD_MODEL')
            api_host = os.getenv('STABILITY_API_HOST')
            api_key = os.getenv('STABILITY_API_KEY')
            if api_key is None:
                raise ValueError("Please set the environment variable API_KEY to your Stability")

            response = requests.post(
                f"{api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "text_prompts": [
                        {
                            "text": "art, " + prompt,
                        }
                    ],
                    "cfg_scale": 7,
                    "height": 512,
                    "width": 512,
                    "samples": 1,
                    "steps": 30,
                }
            )
            if response.status_code != 200:
                raise ValueError(f'{response.text} 第{i+1}页')

            data = response.json()
            if len(data['artifacts']) == 0:
                raise ValueError("image create failed")

            file_path = f'{self.abspath}/images/{i + 1}.png'
            with open(file_path, 'wb') as file:
                file.write(base64.b64decode(data['artifacts'][0]['base64']))
            return file_path

        # 并发生图
        with ThreadPoolExecutor(max_workers=2) as executor:
            image_urls = list(executor.map(generate_image, range(len(self.sd_pages_prompts)), self.sd_pages_prompts))

        if len(image_urls) == 0:
            raise ValueError('images_urls is not empty.')
        self.images_urls = image_urls

        print("图片地址：")
        print(self.images_urls)

        return self.images_urls
