import json

import openai

from util import getenv


class ArticleAnalyzer:
    def __init__(self):
        openai.api_key = getenv("OPENAI_API_KEY")
        openai.organization = getenv("OPENAI_ORGANIZATION")

    def __generate_message(self, content):
        return [
            {
                "role": "user",
                "content": f"""
                문장 : {content}
                
                위의 문장을 바탕으로 요구사항1에 맞추어 요구사항2를 만족하는 결과물을 도출해줘."""
            },
            {
                "role": "system",
                "content": """
                요구사항 1: 주어진 문장에서 핵심 키워드 3가지를 선정해서 결과에 넣어줘
                요구사항 2: 결과는 아래에서 명시한 결과형식과 반드시 똑같은 형태로 만들어줘
                요구사항 3: 결과 형식에 반드시 맞춰야해
                결과형식 :  { "keywords": ["핵심 키워드"] }""",
            }
        ]

    def extract_keywords(self, content):
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=self.__generate_message(content)
        )
        message_content = res.choices[0].message.content
        return json.loads(message_content)['keywords']
