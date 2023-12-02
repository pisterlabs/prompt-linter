import openai
from aiogram import types

from chat_gpt_bot.config import WORKS_CHATS, MODEL, AI_KEY
from chat_gpt_bot.dispatcher import dp


openai.api_key = AI_KEY


def get_ai_answer(question):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": question},
        ]
    )

    return response['choices'][0]['message']['content']


@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def text_reply(message: types.Message):
    if str(message.chat.id) in WORKS_CHATS:
        await message.answer(get_ai_answer(message.text))
    else:
        await message.answer('Invalid chat ID')
