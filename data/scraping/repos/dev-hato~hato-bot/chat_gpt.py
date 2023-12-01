from typing import Optional

from openai import OpenAI

import slackbot_settings as conf

client = OpenAI(api_key=conf.OPENAI_API_KEY)


def chat_gpt(message: str) -> Optional[str]:
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "自然な感じで「鳩は唐揚げ！」という文章を混ぜて発言してください。"},
            {"role": "user", "content": message},
        ],
    )
    return result.choices[0].message.content


def image_create(message: str) -> Optional[str]:
    response = client.images.generate(prompt=message, n=1, size="512x512")
    return response.data[0].url
