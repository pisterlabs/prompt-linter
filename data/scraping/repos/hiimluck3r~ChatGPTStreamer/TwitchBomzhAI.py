import openai
import torch
import sounddevice as sd
import time
import json
import os
from dotenv import load_dotenv, find_dotenv
from transliterate import translit
from num2words import num2words
import twitch_chat_irc

load_dotenv(find_dotenv())
channel_name = os.getenv('channel')
openai.api_key = os.getenv('openai_api_key') # ваш API ключ
connection = twitch_chat_irc.TwitchChatIRC(channel_name, os.getenv('token'))

with open("blacklist.txt", encoding="utf-8", mode="r") as file:
    banwords = [row.strip() for row in file]
print('Bot has started')

# Генерация текста в ответ на полученное сообщение
def gen_text(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=250,
        frequency_penalty=1.2
    )
    response = response['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": response})
    return response, messages


# Генерация голоса
def generate_voice(response):
    sample_rate = 48000
    device = torch.device('cpu')
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='ru',
                              speaker='v3_1_ru')
    model.to(device)
    audio = model.apply_tts(text=response,
                            speaker='aidar',  # aidar, baya, kseniya, xenia, random, eugene
                            sample_rate=sample_rate,
                            put_accent=True,
                            put_yo=True)
    sd.play(audio, sample_rate)
    time.sleep(len(audio) / sample_rate)  # засыпаем на время, пока говорим
    sd.stop()


# Сохраняем паттерн разговора, чтобы использовать его в будущем
def training_save(messages, path):
    with open(path, encoding='utf-8', mode='w') as f_training:
        f_training.writelines(json.dumps(messages, ensure_ascii=False))


# Сохраняем субтитры в .txt для ОБС
def subs_save(content_input, response):
    with open('subs.txt', encoding='utf-8', mode='w') as f_subs:  # в этот файл мы записываем сабы, выводимые в обс
        f_subs.writelines(content_input + '?' + '\n' + response)


# Ищет сообщения в чате Твича, в которых есть слово Комуто и само сообщение не больше 150 символов, включая пробелы
def check_comment():
    data = connection.listen(channel_name, message_limit=1)
    prompt = data[-1]['message']
    user_name = data[-1]['display-name']
    prompt_lower = prompt.lower()
    if 'комуто' in prompt_lower:
        if len(prompt) <= 150:
            return prompt, user_name
        else:
            return '', ''
    else:
        return '', ''

# Приводим числа и английский в слова
def text_normalizer(prompt):
    prompt = translit(prompt, 'ru')
    prompt.replace('+', ' плюс ')
    prompt.replace('=', ' равно ')
    prompt_char = list(prompt)
    i = 0
    number = ''
    while i != len(prompt_char):
        if prompt_char[i].isdigit():
            number = number + prompt_char[i]
        else:
            if number == '':
                pass
            else:
                prompt = prompt.replace(number, num2words(number, lang='ru'))
            number = ''
        i += 1
    print('Normalized:', prompt)
    return prompt

# Главный код, который вызывает остальные функции
def main(content_input, messages, banwords, path):
    content_input = content_input.replace('спой', 'выведи только текст песни')
    messages.append({"role": "user", "content": content_input})
    response, messages = gen_text(messages)
    if any(ele in content_input for ele in banwords):
        content_input = '*' * len(content_input)  # Цензурим банворды. По-хорошему ограничить бы чатботом и тогда эта функция не нужна.
    subs_save(content_input, response)
    training_save(messages, path)
    #print('Response: ' + response) #опционально
    normalized_content_input = text_normalizer(content_input)
    normalized_response = text_normalizer(response)
    generate_voice(normalized_content_input + '? ' + normalized_response + ' ь')  # произносим то, что ввёл пользователь, а также ответ, который дала нейронка
    time.sleep(3)
    with open('subs.txt', encoding='utf-8', mode='w'):
        print('Subs cleared')
    return messages


def debug_input():
    print('DEBUG MODE INPUT')
    content_input = input()
    user_name = input()
    return content_input, user_name

# Основное тело кода, которое отвечает за отлов ошибок
def body():
    content_input, user_name = check_comment()
    #content_input, user_name = debug_input()
    path = 'users_data/'+user_name+'_data.json'
    if content_input != '':
        if os.path.isfile(path):
            try:
                with open (path, encoding='utf-8', mode='r') as f_userData:
                    messages = json.load(f_userData)
                print('Generating text for', user_name)
                main(content_input, messages, banwords, path)
            except openai.error.InvalidRequestError as overflow:
                with open (path, encoding='utf-8', mode='r') as f_userData:
                    messages = json.load(f_userData)
                print('Data overflow: Generating backup')
                with open('users_data/backup_'+user_name+'_data.json', encoding='utf-8', mode='w') as f_backup:
                    f_backup.writelines(json.dumps(messages, ensure_ascii=False))
                print('Writing new user_data...')
                with open('backup_trainingdata_messagelist.json', encoding='utf-8', mode='r') as f_trainingData:
                    basic_instructions = json.load(f_trainingData)  # Вытаскиваем заготовленный обученный базовый датасет.
                    basic_instructions.append(messages[-4])
                    basic_instructions.append(messages[-3])
                    basic_instructions.append(messages[-2])
                    basic_instructions.append(messages[-1])
                with open(path, encoding='utf-8', mode='w') as f_newUserData:
                    f_newUserData.writelines(json.dumps(basic_instructions, ensure_ascii=False))
                with open (path, encoding='utf-8', mode='r') as f_userDataNew:
                    messages = json.load(f_userDataNew)
                print('Generating text for', user_name)
                main(content_input, messages, banwords, path)
            except Exception as e:
                print('!!!!!!!!WARNING!!!!!!!!')
                print(e)
                body()
        else:
            print('Generating file and giving it basic instructions')
            with open(path, encoding='utf-8', mode='w') as f_firstData:
                with open('backup_trainingdata_messagelist.json', encoding='utf-8', mode='r') as f_trainingData:
                    basic_instructions = json.load(f_trainingData)  # Вытаскиваем заготовленный обученный базовый датасет.
                f_firstData.writelines(json.dumps(basic_instructions, ensure_ascii=False))
            messages = basic_instructions
            main(content_input, messages, banwords, path)
            body()
    else:
        print('Blank message accepted. Ignoring')
        body()

while True:
    body()