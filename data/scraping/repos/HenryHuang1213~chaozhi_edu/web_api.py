import json
import os
import time

import openai

from .article_prompt import ArticlePrompt
from .notes_prompt import NotesPrompt

import streamlit as st

class Teacher:

    @staticmethod
    def get_completion(prompt, model="gpt-4"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        openai.organization = "org-YtXb1vm6BeYmYEPTIAG61m59"
        openai.api_key_path = os.path.join(dir_path, '../api.key')
        messages = [{"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    @staticmethod
    def get_text(num):
        num = str(num)
        with open("document/article" + num + ".txt", "r", encoding="utf-8") as f:
            text = f.read()
        return text

    @classmethod
    def get_article_eval(cls, requirements, title, word_count, content):
        start_time = time.time()
        a_prompt = ArticlePrompt(requirements, title, word_count, content)
        prompt = a_prompt.get_prompt()
        # print(prompt)
        response = cls.get_completion(prompt)
        end_time = time.time()

        st.session_state['Evaluation_Cost_Time'] = end_time-start_time
        return response

    @classmethod
    def get_note_eval(cls, content):
        start_time = time.time()
        # return '{}'
        n_prompt = NotesPrompt(content)
        prompt = n_prompt.get_prompt()
        # print(prompt)
        response = cls.get_completion(prompt)
        end_time = time.time()
        st.session_state['Evaluation_Cost_Time'] = end_time-start_time
        return response

# def show_example():
#     content = """
#     今天是星期六，我们一家要去拜访我的姥老姥。我她的印象并不是那么深刻，只记得在小时候见过一次，后来就再也没看到了。
#
#     在路上，阳光明媚。这个季节明明是夏季，早上却跟春天一样。我们还看见了路边五颜六色的花儿，碧绿的草，繁茂的树。
#
#     到了姥姥家后，我第一个冲向的就是姥姥家的后花园，姥姥跟我说我小时候经常在这后花园玩。我到了后花园后，第一个看见的就是菜池，上面种绿油油的蔬菜。我还见了一些水果，最令人注目的是一棵大桃子树，上面都种满了桃子。我拿跑到姥姥那，问姥姥那大桃子能不能吃，姥姥答应了。我立马摘了一了，然后洗了一下，我张开嘴吃了一大口。姥姥老问我甜不甜，我立马说了一句甜。
#
#     直至傍晚，我要走了，临走前，我跟姥姥说：“下次我还来！”“好！”姥姥开心的说了出来
#     """
#     return content
#
# res = Teacher.get_note_eval(show_example())
# print(res)
#
# j_res = json.loads(res)
# print(j_res)


