import sqlite3
import os
import configparser
from openai import OpenAIError
import time

mother_path = os.path.dirname(os.path.dirname(os.getcwd()))

config = configparser.ConfigParser()
config.read(os.path.join(mother_path, 'src/config.ini'))

database = sqlite3.connect(os.path.join(mother_path, 'DataBase/OCAB_DB.db'))
cursor = database.cursor()
reply_ignore = config['Telegram']['reply_ignore'].split('| ')
reply_ignore = list(map(int, reply_ignore))
#print(reply_ignore)

min_token_for_answer = int(config['Openai']['min_token_for_answer'])

# Импорт библиотек

import openai
max_token_count = int(config['Openai']['max_token_count'])

# Создание файла лога если его нет
if not os.path.exists(os.path.join(mother_path, 'src/OpenAI/GPT35turbo/log.txt')):
    with open(os.path.join(mother_path, 'src/OpenAI/GPT35turbo/log.txt'), 'w') as log_file:
        log_file.write('')


def openai_response(message_formated_text):
    # Запуск OpenAI
    # Считаем размер полученного текста
    #print(message_formated_text)
    count_length = 0
    if len(message_formated_text) == 0:
        message_formated_text = [
            {
                "role": "user",
                "content": "Напиши короткий ответ говорящий что контекст сообщения слишком длинный и попроси задать вопрос отдельно без ответа на другие сообщения по ключевому слову"
            }
        ]
    for message in message_formated_text:
        #print(message["content"])
        count_length += int(len(message["content"]))
    #print(count_length)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_formated_text,
            max_tokens=max_token_count - count_length
        )
    except OpenAIError as ex:
        if 'on requests per min. Limit: 3 / min. Please try again' in str(ex):
            response = ('Извини мой процессор перегрелся, дай мне минутку отдохнуть')
        elif 'Bad gateway.' in str(ex):
            response = (
                'Ой, где я? Кажется кто то перерзал мой интернет кабель, подожди немного пока я его починю')
        #запись ошибки в лог с указанием времени и даты
        with open(os.path.join(mother_path, 'src/OpenAI/GPT35turbo/log.txt'), 'a') as log_file:
            log_file.write('\n' + time.strftime("%d.%m.%Y %H:%M:%S") + ' ' + str(ex))
    #Проверка на то что ответ не содержит ошибку
    count_length = 0

    return response

def sort_message_from_user(message_formated_text, message_id):
    # print(int(*(
    #    cursor.execute("SELECT message_sender FROM message_list WHERE message_id = ?", (message_id,)).fetchone())))
    if int(*(
            cursor.execute("SELECT message_sender FROM message_list WHERE message_id = ?",
                           (message_id,)).fetchone())) == 0:
        message_formated_text.append({
            "role": "assistant",
            "content": str(*(cursor.execute("SELECT message_text FROM message_list WHERE message_id = ?",
                                            (message_id,)).fetchone()))
        })
    else:
        message_formated_text.append({
            "role": "user",
            "content": str(*(cursor.execute("SELECT message_text FROM message_list WHERE message_id = ?",
                                            (message_id,)).fetchone()))
        })
    #Проверка что длина всех сообщений в кортеже не превышает max_token_count-min_token_for_answer
    return message_formated_text

def openai_collecting_message(message_id, message_formated_text):
    # собирает цепочку сообщений для OpenAI длинной до max_token_count
    # проверяем что сообщение отвечает на другое сообщение
    #print(int(*(cursor.execute("SELECT answer_id FROM message_list WHERE message_id = ?", (message_id,)).fetchone())))
    #print(reply_ignore)
    if int(*(cursor.execute("SELECT answer_id FROM message_list WHERE message_id = ?", (message_id,)).fetchone())) not in reply_ignore:
        # Продолжаем искать ответы на сообщения
        #print(int(*(cursor.execute("SELECT answer_id FROM message_list WHERE message_id = ?", (message_id,)).fetchone())))
        message_formated_text = openai_collecting_message(int(*(cursor.execute("SELECT answer_id FROM message_list WHERE message_id = ?", (message_id,)).fetchone())), message_formated_text)
        #Проверяем ID отправителя сообщения, если 0 то это сообщение от бота
        sort_message_from_user(message_formated_text, message_id)
    else:
        # Проверяем ID отправителя сообщения, если 0 то это сообщение от бота
        sort_message_from_user(message_formated_text, message_id)
    return message_formated_text


def openai_message_processing(message_id):
    #проверяем на наличие сообщения в базе данных
    if cursor.execute("SELECT message_text FROM message_list WHERE message_id = ?", (message_id,)).fetchone() is None:
        return None
    else:
        # проверяем на то что сообщение влезает в max_token_count с учётом message_formated_text
        message_formated_text = [
            {
                "role": "system",
                "content": config['Openai']['story_model']
            }
        ]
        if ((len(str(cursor.execute("SELECT message_text FROM message_list WHERE message_id")))) < (max_token_count - len(message_formated_text[0]['content']))):
            message_formated_text = openai_collecting_message(message_id, message_formated_text)
            count_length = 0
            # Обработка невозможности ответить на сообщение
            try:
                for message in message_formated_text:
                    count_length += len(message['content'])
                while count_length > max_token_count - min_token_for_answer:
                    message_formated_text.pop(1)
                    count_length = 0
                    for message in message_formated_text:
                        count_length += len(message['content'])
            except IndexError:
                message_formated_text = [
            {
                "role": "system",
                "content": "Выведи сообщение об ошибке."
            }
        ]
            response = openai_response(message_formated_text)
            return response
        else:
            return f"Сообщение слишком длинное, максимальная длина сообщения \
            {max_token_count - len(message_formated_text[0]['content'])} символов, укоротите его на \
            {len(str(cursor.execute('SELECT message_text FROM message_list WHERE message_id'))) - max_token_count} символов"