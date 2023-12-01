import os
import aiohttp
import datetime as dt
from aiogram import types
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Необходимо задать переменную окружения OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

async def generate_dalle_image(prompt: str):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="2048x2048",
            quality="high",
            n=1,
        )
    except Exception as e:
        raise Exception(f"Ошибка при генерации изображения: {e}")

    image_url = response.data[0].url

    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                raise Exception(f"Не удалось загрузить изображение. Код статуса: {resp.status}")
            image_data = await resp.read()

    filename = f"dalle-{dt.datetime.now().strftime('%d-%m-%Y')}.png"
    with open(filename, "wb") as f:
        f.write(image_data)
    return filename

async def dalle_command_handler(message: types.Message):
    prompt = message.get_args()
    if not prompt:
        await message.reply("Пожалуйста, укажите промпт после команды /dalle.")
        return

    await message.reply("Генерация изображения началась, это может занять некоторое время...")
    try:
        image_file = await generate_dalle_image(prompt)
        with open(image_file, "rb") as photo:
            await message.reply_photo(photo, caption="Ваше изображение готово.")
    except Exception as e:
        await message.reply(str(e))
    finally:
        if os.path.exists(image_file):
            os.remove(image_file)