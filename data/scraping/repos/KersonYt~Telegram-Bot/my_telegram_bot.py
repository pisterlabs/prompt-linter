import logging
import openai
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram import Bot

API_KEY = "tuApiDelBot"

openai.api_key = "tuApiDeOpenAi"

messages = [{"role": "system", "content": "You are a kind helpful assistant"}]

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def call_ChatGPT(message):
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages 
        )
         
    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply;
    
def remove_first_word(s):
    words = s.split()
    if len(words) > 1:
        return ' '.join(words[1:])
    else:
        return '' 

def handle_text(update: Update, context: CallbackContext):
    message = update.message.text
    chat_id = update.message.chat_id
    if message.lower().startswith("dibuja"):
        processing_message = context.bot.send_message(chat_id=chat_id, text="Dibujando...")
        try:
            response = openai.Image.create(
                prompt=remove_first_word(message),
                n=1,
                size="1024x1024"
            )
            reply = response['data'][0]['url']
        except:
            reply = "Error llamando al servicio de Image"
            print(reply)
    else:
        try:
            processing_message = context.bot.send_message(chat_id=chat_id, text="Pensando...")
            reply = call_ChatGPT(message)
        except:
            reply = "Error llamando al servicio de ChatCompletion"
            print(reply)
        
    context.bot.send_message(chat_id=chat_id, text=reply)

def main():
    updater = Updater(token=API_KEY, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, handle_text))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
