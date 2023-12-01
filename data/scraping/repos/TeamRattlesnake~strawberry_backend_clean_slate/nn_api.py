"""
Модуль с реализацией общения с апи нейросетей
"""

import openai

MAX_WORDS_LEN = 3000
MIN_WORDS_LEN = 5
SYSTEM_PROMPT = "Тебя зовут Strawberry, ты помогаешь писать посты в сообщества социальных сетей. Ты должен отвечать одним текстом поста для публикации"
NO_SOURCE_TEXTS_REPLACEMENT = (
    "Старых постов в сообществе нет, так что придумай что-то креативное"
)
OLD_TEXTS_PLACEHOLDER = "[OLD_TEXTS]"
HINT_PLACEHOLDER = "[HINT]"


class NNException(Exception):
    """
    Класс исключения, связанного с подготовкой данных и
    отправкой запроса на апи нейросетей
    """

    pass


class NNApi:
    """
    Класс для подготовки запросов и общения с API нейросети
    """

    def __init__(self, token: str):
        openai.api_key = token
        self.context = ""
        self.query = ""
        self.result = ""

    def load_context(self, path: str):
        """
        Загружает шаблон контекста из файлика
        """
        try:
            with open(path, "r", encoding="UTF-8") as ctx_file:
                self.context = ctx_file.read()
        except Exception as exc:
            raise NNException(f"Error in load_context: {exc}") from exc

    def prepare_query(self, context_data: list[str], hint: str):
        """
        Расставляет данные по шаблону контекста
        """
        try:
            if len(hint) >= MAX_WORDS_LEN:
                raise NNException(
                    "Error in prepare_query: the request is too long (hint alone is larger than allowed input in model)"
                )

            source_texts_string = ""

            for text in context_data:
                # Собираем строку с постами чтобы она была не длиннее, чем нужно.
                # Считаю не количество слов, а количество букв потому что
                # токенизатор не любит русский
                if (
                    len(source_texts_string) + len(text) + len(hint) + len(self.context)
                ) >= MAX_WORDS_LEN:
                    continue
                source_texts_string += f"{text}\n\n"

            if (
                len(source_texts_string) > MIN_WORDS_LEN
            ):  # Минимальная проверка на валидность контекста
                self.query = self.context.replace(
                    OLD_TEXTS_PLACEHOLDER,
                    source_texts_string,
                )
            else:
                # Если контекст слишком маленький,
                # то надо просто сказать нейросети быть креативной
                self.query = self.context.replace(
                    OLD_TEXTS_PLACEHOLDER,
                    NO_SOURCE_TEXTS_REPLACEMENT,
                )

            self.query = self.query.replace(HINT_PLACEHOLDER, hint)
            self.query = self.query.strip()
        except Exception as exc:
            raise NNException(f"Error in prepare_query: {exc}") from exc

    def send_request(self):
        """
        Отправляет запрос к API нейросети
        """
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self.query},
                ],
            )
            self.result = completion.choices[0].message.content
        except Exception as exc:
            raise NNException(f"Error in send_request: {exc}") from exc

    def get_result(self) -> str:
        """
        Возвращает ответ нейросети
        """
        return self.result
