import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

token = 'Token from bot father'
openai.api_key = 'API key openai'

bot = Bot(token)
dp = Dispatcher(bot)

@dp.message_handler()
async def send(message: types.Message):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message.text,
        temperature=0.9,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )
    await message.answer(response['choices'][0]['text'])

executor.start_polling(dp, skip_updates=True)
