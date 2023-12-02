from pyrogram import Client
from pyrogram import filters
from data import config
import textgen.chatgpt as tg
import openai


# TODO: Save variables to config
app = Client("my_account", config.telegram_app_id, config.telegram_app_hash)


@app.on_message(filters.regex("^\?") & ~filters.me)
async def say_wise(client, message):
    openai.api_key = config.next_token()
    mes = message.text
    await app.send_message(message.chat.id, await tg.generate_reply(mes), reply_to_message_id=message.id)

app.run()
