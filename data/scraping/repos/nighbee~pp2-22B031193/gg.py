import openai
from aiogram import Bot, Dispatcher, executor, types

botToken = '6277072307:AAHk63pZ2zj-oGMkT9kTLtngaU9cnn2FTik'
openAi = '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'

bot = Bot(token=botToken)
dp = Dispatcher(bot)

async def welcome(message: types.Message):
    await message.reply('Hi, how can I help?')

async def generate_response(prompt: str) -> str:
    openai.api_key = openAi
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

async def echo(message: types.Message):
    prompt = message.text
    response = await generate_response(prompt)

    await message.answer(response)

if __name__ == '__main__':
    dp.register_message_handler(welcome, commands=['start'])
    dp.register_message_handler(echo)
    executor.start_polling(dp, skip_updates=True)
