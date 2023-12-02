import sys

sys.path.append("..")

import openai  # noqa
from app.config import ProdSettings  # noqa


class ChatGPT:
    """
    Работа с ChatGPT
    """

    def __init__(self):
        self.settings = ProdSettings()
        openai.api_key = self.settings.OPENAI_KEY

    def send_request(self, prompt: str) -> str:
        """
        Отправляет запрос в ChatGPT и возвращает ответ
        :param prompt: Запрос
        :return:
        """
        return (
            openai.ChatCompletion.create(
                model=self.settings.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            .choices[0]
            .message.content
        )
