import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

token = '6268136991:AAFClP5dM0d_oV5QOMWV4JrXjlZFRJYLmOM'
openai.api_key = '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'

bot = Bot(token)
dp = Dispatcher(bot)

# print(openai.Model.list())

users = {5016528260, 517748480}

accepted_users = lambda message: message.from_user.id not in users


@dp.message_handler(accepted_users, content_types=['any'])
async def handle_unwanted_users(message: types.Message):
    await message.answer(
        "Kechirasiz, bot faqat tasdiqlangan foydalanuvchilar uchun ishlaydi. Agar siz xuddi shu botni yozmoqchi bo'lsangiz - havolaga o'ting: https://t.me/egamberdoyevsardor")
    return


max_symbols = lambda message: int(len(message.text)) > 2000


@dp.message_handler(max_symbols, content_types=['any'])
async def handle_unwanted_users(message: types.Message):
    await message.answer(
        "Xato! Kiritilgan belgilar soni maksimal 2000 qiymatdan oshib ketdi \n\nKiritilgan belgilar soni: " + str(
            len(message.text)) + "\n\nSo'rovingizni qisqartiring \n\nOpenAI API tokenlari haqida koʻproq oʻqing: https://t.me/egamberdoyevsardor")
    return


@dp.message_handler()
async def send(message: types.Message):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message.text,
        temperature=0.9,
        max_tokens=4000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    await message.answer(response['choices'][0]['text'])


executor.start_polling(dp, skip_updates=True)
