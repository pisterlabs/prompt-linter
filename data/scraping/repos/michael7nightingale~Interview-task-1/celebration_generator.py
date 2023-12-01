"""
Функционал ChatGTP API для генерации поздравлений.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import openai
from abc import ABC, abstractmethod

from data_.settings import CELEBRATION_GENERATOR, logger
from src import exceptions


eventor = threading.Event()


class BaseCelebrator(ABC):
    def __init__(self, token: str):
        self.__token = token
        openai.api_key = self.__token
        logger.info("__CELEBRATION_GENERATOR__ Успешно инициализирован и авторизирован API")



    @staticmethod
    def replace_message_with_data(name: str, birthday: str) -> str:
        """Заменяет служебные пропуски в шаблоне запроса на данные"""
        return CELEBRATION_GENERATOR['MESSAGE'].replace('__name__', name).replace('__birthday__', birthday)

    @staticmethod
    @abstractmethod
    def chat(message: str) -> str:
        pass

    @abstractmethod
    def generate_celebrations(self, data: list[dict],
                              name_column: str,
                              birthday_column: str) -> list[str]:
        pass


class Celebrator(BaseCelebrator):
    """Класс ПОЗДРАВЛЯТОР-3000 для Chat-GTP API."""

    @staticmethod
    def chat(message: str) -> str:
        """Функция общения с API Chat-GPT. Единственный вопрос - единственный ответ."""
        print(1, message)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=message,
            max_tokens=2048,
            n=1,
            temperature=0.5,
        )
        logger.info(f"__CELEBRATION_GENERATOR__ Ответ от CHAT_GPT(синхронно): {response.choices[0]['text']}")
        return response.choices[0]['text']

    def generate_celebrations(self, data: list[dict],
                              name_column: str,
                              birthday_column: str) -> list[str]:
        """Генерация списка поздравлений"""
        # проверка на валидность формата представления данных
        if all(name_column in line and birthday_column in line for line in data):
            args = [self.replace_message_with_data(str(person['name']), str(person['birthday'])) for person in data]
            return [self.chat(msg) for msg in args]

        else:
            logger.error(f"__CELEBRATION_GENERATOR__ {exceptions.InvalidDataFormat.__doc__}: названия колонок не соответствуют конфигурации")
            raise exceptions.InvalidDataFormat


class AsyncCelebrator(BaseCelebrator):
    """Асинхронный класс, то есть запросы к API происходят параллельно в рамках подзадач."""

    @staticmethod
    def chat(message: str) -> str:
        """Функция общения с API Chat-GPT. Единственный вопрос - единственный ответ."""
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Сделай поздравление человеку с именем {message}. Добавь в него юмор",
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )
        logger.info(f"__CELEBRATION_GENERATOR__ Ответ от CHAT_GPT(aсинхронно): {response.choices[0]['text']}")
        # из-за проблем с кодировкой пришлось обработать строку именно так
        return response.choices[0]['text'].replace('\n', '').replace('\c', '')[1:]

    def generate_celebrations(self, data: list[dict],
                                    name_column: str,
                                    birthday_column: str) -> list[str]:
        """Генерация списка поздравлений"""
        # проверка на валидность формата представления данных
        if all(name_column in line and birthday_column in line for line in data):
            args = [self.replace_message_with_data(str(person['name']), str(person['birthday'])) for person in data]
            pool = ThreadPoolExecutor()
            result = pool.map(self.chat, args)
            return [i for i in result]
        else:
            logger.error(f"__CELEBRATION_GENERATOR__ {exceptions.InvalidDataFormat.__doc__}: названия колонок не соответствуют конфигурации")
            raise exceptions.InvalidDataFormat


class ThreadCelebrator(BaseCelebrator):

    def generate_celebrations(self, data: list[dict],
                              name_column: str,
                              birthday_column: str) -> list[str]:
        tasks = []
        for i in data:
            tasks.append(
                threading.Thread(
                    target=self.chat, args=(self.replace_message_with_data(str(i['name']), str(i['birthday'])), )
                )
                         )
            tasks[-1].start()

        return tasks

    @staticmethod
    def chat(message: str) -> str:
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=message,
            max_tokens=2048,
            n=1,
            temperature=0.5,

        )
        eventor.set()
        return response.choices[0]['text']





if __name__ == '__main__':
    c = Celebrator(CELEBRATION_GENERATOR['TOKEN'])
    print(c.chat('Напиши поздравление для друга Матвея в его день рождения, который родился 15.10.\
     Сделай это душевно и лаконично, всё таки это день рождения.'))

