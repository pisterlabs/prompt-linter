from telegram.ext import Updater, MessageHandler, Filters
import telegram
import openai

openai.api_key = "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"
TELEGRAM_API_TOKEN = "6623406817:AAE4HF9IR-qtM5xOS9302jn5b9ubmfAdqFY"

messages = [{"role": "system", "content": "You are TelegramGPT, a helpful telegram bot that is always concise and polite in its answers."}]


def text_message(update, context):
    messages.append({"role": "user", "content": update.message.text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    update.message.reply_text(text=f"*[Bot]:* {ChatGPT_reply}", parse_mode=telegram.ParseMode.MARKDOWN)
    messages.append({"role": "assistant", "content": ChatGPT_reply})


updater = Updater(TELEGRAM_API_TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), text_message))
updater.start_polling()
updater.idle()