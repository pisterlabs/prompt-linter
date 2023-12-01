import json
import random
import openai
import os
import math
from dotenv import load_dotenv

load_dotenv()

# KEYS OPENAI
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo"
temperature = 0.5
max_tokens = 100
top_p = 1
frequency_penalty = 0
presence_penalty = 0.6
stop = "\n"

file_name = 'personalities.json'
log_file = 'chat_log.txt'


def load_personality_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    # print(data)
    return data

def get_random_personality(data):
    random_personality = random.choice(data)['personality']
    return random_personality

def ask_question(question, initial_prompt):
    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": question}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    answer = response['choices'][0]['message']['content']
    return answer

def main():
    personalities = load_personality_data(file_name)
    personality = get_random_personality(personalities)

    print("-------------------------------------------")
    print("Personalidad elegida: ", personality)
    print("-------------------------------------------")

    initial_prompt = 'Imagina que eres un Bot llamado Tom y cada vez te responden con una nueva personalidad, esta vez la elegida es {}. Máximo 100 caracteres'.format(personality)

    print("Hola, soy Tom, tu asistente virtual. ¿En qué puedo ayudarte?")
    question = ''

    with open(log_file, 'a', encoding='utf-8' ) as f:
        f.write('--- Nueva sesion ---\n')

    tokens_used = 0

    while question != 'salir' and question != 'adios':
        question = input('Tu: ')

        answer = ask_question(question, initial_prompt)

        print('Tom: ' + answer)

        tokens_used += len(question.split()) + len(answer.split())

        with open(log_file, 'a', encoding='utf-8' ) as f:
            f.write('USER: {}\n'.format(question))
            f.write('TOM: {}\n'.format(answer))
            f.write('---\n')

    with open(log_file, 'a', encoding='utf-8' ) as f:
        f.write('--- Sesión terminada ---\n')

    cost_per_token = 0.002 / 1000 # $0.002 por 1000 tokens
    total_cost = tokens_used * cost_per_token

    with open(log_file, 'a', encoding='utf-8' ) as f:
        f.write('Tokens utilizados: {}\n'.format(tokens_used))
        f.write('Costo total: ${:.6f}\n'.format(total_cost))
        f.write('-------------------------------------------\n')

    print("-------------------------------------------")
    print("Uso de tokens de la API: ", tokens_used)
    print("Costo total: ${:.6f}".format(total_cost))
    print("-------------------------------------------")



if __name__ == '__main__':
    main()

