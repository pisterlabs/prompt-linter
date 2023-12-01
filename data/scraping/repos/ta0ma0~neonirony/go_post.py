import openai
import time
import os
import random
import string
import sys
import json
import requests

def clear_screen():
    # Очистка экрана
    os.system('cls' if os.name == 'nt' else 'clear')

def print_random_ascii(length=67):
    # ANSI escape codes для различных цветов
    colors = [
        "\033[91m",  # Красный
        "\033[92m",  # Зеленый
        "\033[93m",  # Желтый
        "\033[94m",  # Синий
        "\033[95m",  # Пурпурный
        "\033[96m",  # Голубой
        "\033[97m"   # Белый
    ]
    # Выбор случайного цвета
    random_color = random.choice(colors)

    # Генерация случайной строки
    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    # Печать строки с выбранным цветом
    print(random_color + random_chars + "\033[0m")  # \033[0m сбрасывает форматирование


def get_gpt_advice(api_key, prompt):
    # Получение ответа от GPT
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=800
    )

    return response.choices[0].text.strip()

def get_dalle_prompt(api_key, advice):
    # Получение ответа от GPT
    openai.api_key = api_key
    preprompt = "Create a request for Dall-e 3 on English languge from the one that comes after [prompt], the generated image should have an illustration of what is in the main message, the image should be in cyberpunk style"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=preprompt + "[prompt]" + advice,
        max_tokens=256
    )

    dalle_prompt = response.choices[0].text.strip()
#    print(dalle_prompt)
    return dalle_prompt


def _write_to_file(advice, dirname='.'):
    with open(f'{dirname}/advice.txt', 'w') as f:
        f.write(advice + '\n')

def create_image(api_key, prompt):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    data = json.dumps({
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    })

    response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        print("Something went wrong")
        return None

def save_image(image_data, dirname):
    if image_data:
        image_url = image_data['data'][0]['url']
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            with open(f'{dirname}/image.png', 'wb') as f:
                f.write(image_response.content)

def main(api_key, prompt):
    while True:
        start_time = time.time()
        clear_screen()
        api_key = os.environ.get("OPENAI_API_KEY", None)
        advice = get_gpt_advice(api_key, prompt)
        print(advice)
        _write_to_file(advice)
        user_input = input('Continue? Y/n/p').strip().lower()
        if user_input == 'n':
            sys.exit()
        elif user_input == 'y' or user_input == '':
            # Продолжить выполнение программы
            pass
        elif user_input == 'p':
            dirname = 'post_' + ''.join(random.choices(string.ascii_letters + string.digits, k=5))
            os.makedirs(dirname, exist_ok=True)
            dalle_prompt = get_dalle_prompt(api_key, advice)
            image_data = create_image(api_key, dalle_prompt)
            print(dalle_prompt)
            save_image(image_data, dirname)
            _write_to_file(advice + '\n\n' +  dalle_prompt, dirname)

        while True:
            current_time = time.time()
            if current_time - start_time > 1700:  # Проверка времени на превышение 1700 секунд
                break
            time.sleep(3)  # Пауза на 1 секунду для обновления текущего времени
            print_random_ascii()

# Пример использования:
api_key = os.environ.get("OPENAI_API_KEY", None)
#prompt = "Напиши историю в стиле киберпанк, в которой есть признаки этого жанра, используй места и имена из известных произведений киберпанка, добавь хакерского жаргона, где это уместно"
prompt = "Напиши короткую историю (400-500 символов) в стиле киберпанк, включающую элементы жанра, места и имена из известных произведений, а также хакерский жаргон. Сделай акцент на краткость и лаконичность."
main(api_key, prompt)

