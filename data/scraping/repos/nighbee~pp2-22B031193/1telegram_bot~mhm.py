import openai
from aiogram import Bot, Dispatcher, executor, types

botToken = '5965750764:AAGzp7lxFO0Se2s_uBYmYL6LiOL_iRx-Jg4'
openAi = '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'

bot = Bot(token=botToken)
dp = Dispatcher(bot)

async def welcome(message: types.Message):
    await message.reply('Hi, how can I help?')

async def generate_response(prompt: str) -> str:
    openai.api_key = openAi
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

async def echo(message: types.Message):
    prompt = f"User: {message.text}\nAI:"
    response = await generate_response(prompt)

    await message.answer(response)

if __name__ == '__main__':
    dp.register_message_handler(welcome, commands=['start'])
    dp.register_message_handler(echo)
    executor.start_polling(dp, skip_updates=True)
