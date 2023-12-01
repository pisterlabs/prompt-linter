from telegram.ext import Updater, MessageHandler, Filters
import telegram
import openai
from moviepy.editor import AudioFileClip

openai.api_key = "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"
TELEGRAM_API_TOKEN = "6623406817:AAE4HF9IR-qtM5xOS9302jn5b9ubmfAdqFY"

messages = [{"role": "system", "content": "You are Chitti the robot,a helpful telegram bot who is always rude in its answers and also trolls the user,introduce yourself everytime a convo starts as  I am chitti,reloaded version 2.0,i will set your sceens on fire!! "}]

def text_message(update, context):
    messages.append({"role": "user", "content": update.message.text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    update.message.reply_text(text=f" {ChatGPT_reply}", parse_mode=telegram.ParseMode.MARKDOWN)
    messages.append({"role": "assistant", "content": ChatGPT_reply})

def voice_message(update, context):
    update.message.reply_text("I've received a voice message! Please give me a second to respond :)")
    voice_file = context.bot.getFile(update.message.voice.file_id)
    voice_file.download("voice_message.ogg")
    audio_clip = AudioFileClip("voice_message.ogg")
    audio_clip.write_audiofile("voice_message.mp3")
    audio_file = open("voice_message.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text
    update.message.reply_text(text=f"*[You]:* _{transcript}_", parse_mode=telegram.ParseMode.MARKDOWN)
    messages.append({"role": "user", "content": transcript})
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
dispatcher.add_handler(MessageHandler(Filters.voice, voice_message))
updater.start_polling()
updater.idle()