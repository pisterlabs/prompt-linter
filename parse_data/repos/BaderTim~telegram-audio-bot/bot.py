import logging
import time
from telegram import Update
from openai import OpenAI
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import os
from dotenv import load_dotenv
load_dotenv(".env")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

os.makedirs("./temp_audio_files", exist_ok=True)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="Hi 👋\nIch kann Audiodateien für dich transkribieren. Sende mir hierfür einfach eine Audiodatei - und ich erledige den Rest."
    )

async def received_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_size_in_MB = update.message.voice.file_size/(1024*1024)
    await transcribe(update, context, update.message.voice.duration, file_size_in_MB)

async def received_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_size_in_MB = update.message.audio.file_size/(1024*1024)
    await transcribe(update, context, update.message.audio.duration, file_size_in_MB)


async def transcribe(update: Update, context: ContextTypes.DEFAULT_TYPE, duration: int, file_size_in_MB: float):
    # Run checks
    if file_size_in_MB > 25:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=f"Die Audiodatei ist zu groß. Bitte sende mir eine Datei, die kleiner als 25 MB ist. Deine Datei ist {round(file_size_in_MB, 2)} MB groß."
        )
        return
    # Download file
    new_file = await update.message.effective_attachment.get_file()
    new_file_name = f"./temp_audio_files/{new_file.file_id}.mp3"
    await new_file.download_to_drive(new_file_name)
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=f"Audiodatei wurde empfangen.\n➤ Dauer: {time.strftime('%M:%S', time.gmtime(duration))}\n➤ Größe: {round(file_size_in_MB, 2)} MB\n➤ Kosten: {round(duration/60*0.6,2)} ct\n\nTranskribiere..."
    )
    # Transcribe
    file_to_transcribe = open(new_file_name, "rb")
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            prompt="",
            file=file_to_transcribe,
            language="de",
            response_format="text"
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=f"Es ist ein Fehler aufgetreten. Bitte versuche es erneut.\n\nFehlermeldung: {e}"
        )
        file_to_transcribe.close()
        os.remove(new_file_name)
        return
    # Send transcription
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=f"{transcript}"
    )
    # Delete file
    file_to_transcribe.close()  
    os.remove(new_file_name)

if __name__ == "__main__":
    application = ApplicationBuilder().token(os.environ.get("TELEGRAM_BOT_API_KEY")).build()
    
    start_handler = CommandHandler("start", start)
    audio_handler = MessageHandler(filters.AUDIO & (~filters.COMMAND), received_audio)
    voice_handler = MessageHandler(filters.VOICE & (~filters.COMMAND), received_voice)

    application.add_handler(start_handler)
    application.add_handler(audio_handler)
    application.add_handler(voice_handler)
    
    application.run_polling()
