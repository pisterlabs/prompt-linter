import os
import discord
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def cutti_bot():
    user_input = input(f'YMC: ')
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_input,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0.6, )

    cutti = response["choices"][0]["text"]
    print(f'Aniki: {cutti}')


def franki_bot(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6, )

    message = response["choices"][0]["text"]
    return message

