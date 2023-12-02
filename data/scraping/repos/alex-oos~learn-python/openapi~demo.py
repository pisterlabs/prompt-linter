#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    @Author : tangkaize
#
#              _____               ______
#     ____====  ]OO|_n_n__][.      |    |]
#    [________]_|__|________)<     
#     oo    oo  'oo OOOO-| oo\_   ~o~~~o~'
# +--+--+--+--+--+--+--+--+--+--+--+--+--+
#    @Time : 2022/12/17 11:46
#    @FIle： tools.py
#    @Software: PyCharm
#    @description: chatgpt demo

# openai.organization = "org-H6bImdZRd5oLO8ziWy1iVy8W"
# openai.api_key = os.getenv("")
# openai.Model.list()

import openai

openai.api_key = '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'


def chat_gpt(prompt):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=0.8,
        max_tokens=120,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.0,
    )
    print(response.choices[0].text)


if __name__ == '__main__':
    while True:
        print('请输入你想要的内容：')
        prompt = input()
        chat_gpt(prompt)
