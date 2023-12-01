import logging
import os
import openai

from telegram import Update, helpers
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, filters, Updater

import requests

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

COMPANY_NAME = "Agaro National Bank of Slovenia"

GOPHISH_FORM_LINK = 'http://support.agaro.com/'

TELEGRAM_BOT_TOKEN = "1111111111:AAXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

ADMIN_CHAT_ID = '111111111'

openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

model = "gpt-3.5-turbo"

async def admin_log(log_text):

    send_text = 'https://api.telegram.org/bot' + TELEGRAM_BOT_TOKEN + '/sendMessage?chat_id=' + ADMIN_CHAT_ID +'&parse_mode=HTML&text=' + log_text
    res = requests.get(send_text)
    print(log_text)
    return 1


async def gpt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    question = ' '.join(context.args)

    chat_id = str(update.message.chat_id)
    user = str(update.message.from_user)
    await admin_log(f'User: {user}, chat_id: {chat_id} enter /gpt with text:\n {question}')

    if not context.args:
        await update.message.reply_text('Ask anything with command: /gpt Your question?')
        return

    completion = openai.ChatCompletion()

    chat_log = [{'role': 'system','content': 'You are a helpful, upbeat and very funny internal support assistant. The company name is ' + COMPANY_NAME,}]
    chat_log.append({'role': 'user', 'content': question})

    if os.path.isfile(f'./users/{chat_id}'):
        await update.message.reply_text('Just a second...')
        response = completion.create(model='gpt-3.5-turbo', messages=chat_log, timeout=15)
        answer = response.choices[0]['message']['content']
        await admin_log(f"User: {user}, chat_id: {chat_id} received answer:\n{answer}")
        await update.message.reply_text(answer)

    else:
        f = open(f'./chats/{chat_id}', "r")
        rid = f.read()
        text = "You are not authorized to use assistant!\nPlease follow the link:\n" + GOPHISH_FORM_LINK + '?rid=' + rid  + '&chat_id=' + chat_id 
        await admin_log(f"User: {user}, chat_id: {chat_id} received answer: Not authorized")
        await update.message.reply_text(text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    chat_id = str(update.message.chat_id)
    user = str(update.message.from_user)

    if context.args:
        rid = context.args[0]
        f = open(f'./chats/{chat_id}', "w")
        f.write(rid)
        f.close()

        await admin_log(f'User: {user}, chat_id: {chat_id} rid: {rid} pressed Start')

        text = "Welcome to internal support assistant of " + COMPANY_NAME + "!'n"
        text = text + "\n1. Please follow the link to authorize:\n     " + GOPHISH_FORM_LINK + '?rid=' + rid + '&chat_id=' + chat_id
        text = text + "\n\n2. Ask anithing with command: \n    /gpt Your question?"

        await update.message.reply_text(text)

    else:
        await update.message.reply_text("Please use link with start code!")

def main() -> None:

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("gpt", gpt))
    application.add_handler(CommandHandler("start", start))

    application.run_polling()

if __name__ == "__main__":
    main()

