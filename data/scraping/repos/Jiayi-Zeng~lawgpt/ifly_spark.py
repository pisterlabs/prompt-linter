# -*- coding: utf-8 -*-

import openai
import json


def Spark(query, model='gpt-3.5-turbo'):
    try:
        openai.api_key = 'your api key'
        openai.proxy = {'http': '127.0.0.1:7890', 'https': '127.0.0.1:7890'}
        # openai.proxy = {'http': '124.70.203.196:20171', 'https': '124.70.203.196:20171'}
        messages = [
            {"role": "user", "content": query},
        ]
        print(model)
        print(messages)

        response = openai.ChatCompletion.create(model=model, messages=messages,temperature=0)
        answer = response.choices[0].message.content
        answer = json.dumps(answer, ensure_ascii=False)
        return answer
    except Exception as e:
        print('报错：',str(e))


if __name__ == '__main__':
    query = '你是谁'
    res = Spark(query)
    print(res)