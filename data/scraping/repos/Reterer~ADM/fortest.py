import requests
import os
import openai
import json
import summorize.config as config
import re


openai.organization = config.organization
openai.api_key = config.api_key

control_prompt = """
Вот шаблон объявления c вакансиями в формате HTML.
<!-- Если в объявлении нет должности, то вывести "не указана" --!>
<h2>Должность:</h2>
Название вакансии<br>

<!-- Если в объявлении нет компании, то вывести "не указана" --!>
<h2>Компания:</h2>
Название компании<br>
<!-- Если в объявлении нет зарплаты, тогда оставить "не указана" --!>
<h2>Зарплата:</h2>
в рублях<br>

<!-- Тут должны быть требования к кандидату, если нет требований, тогда написать, что рады видеть всех -->
<h2>Требования:</h2>
<ul>
  <li>Требование 1</li>
  <li>Требование 2</li>
  <li>Требование 3</li>
</ul>

<!-- Тут надо написать, что нужно будет делать на работе -->
<h2>Задачи:</h2>
<ul>
  <li>Задача 1</li>
  <li>Задача 2</li>
  <li>Задача 3</li>
</ul>

Следующее объявление нужно сократить, вычленить только нужное и переделать под формат выше. Твой ответ должен обязательно быть по формату, больше никаких лишних полей. Нельзя выдумывать новую информацию и вывести результат в HTML коде(Это обязательно):

"""


def translate(text, target, source):
    body = {
        "sourceLanguageCode": source,
        "targetLanguageCode": target,
        "texts": text,
        "folderId": config.folder_id,
    }

    headers = {"Content-Type": "application/json", "Authorization": config.Ya_api_key}

    # Перевод
    response = requests.post(
        "https://translate.api.cloud.yandex.net/translate/v2/translate",
        json=body,
        headers=headers,
    )

    data = json.loads(response.text)
    text = data["translations"][0]["text"]
    return text


def gpt_request(text):
    model_engine = "text-davinci-003"

    # Формирование анкеты
    response = openai.Completion.create(
        engine=model_engine,
        prompt=text,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0,
        top_p=0.1,
        logprobs=0,
    )
    response = response.choices[0].text
    return response


def summaryVacancy(vacancy_text):
    prompt = control_prompt + vacancy_text
    translated_promnt = translate(prompt, "en", "ru")
    gpt_answer_en = gpt_request(translated_promnt)
    gpt_answer_ru = translate(gpt_answer_en, "ru", "en")
    modified_text = re.sub(r"<ул>", "<ul>", gpt_answer_ru, flags=re.IGNORECASE)
    # print(modified_text)
    return modified_text
