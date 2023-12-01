from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

import openai

import configparser

config = configparser.ConfigParser()
config.read('config.cfg')

TOKEN = config['API_KEYS']['TOKEN']
OPENAI_KEY = config['API_KEYS']['OPENAI_KEY']

CHOOSING, TYPING_PROMPT = range(2)

def build_keyboard():
    keyboard = [['PUSH ME AND THEN...']]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

def start(update: Update, context: CallbackContext) -> int:
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi, {user.mention_markdown_v2()}\. This is dummy Telegram bot for ChatGPT\. Please click the button below to start\.',
        reply_markup=build_keyboard(),
    )
    return CHOOSING

def want_start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Type your request to ChatGPT!")
    return TYPING_PROMPT

def user_prompt(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("ChatGPT is thinking...")

    user_prompt_text = update.message.text
    response = chatgpt_send(user_prompt_text)
    update.message.reply_text(f'{response}')
    update.message.reply_text("Press this button to send one more request to ChatGPT", reply_markup=build_keyboard())

    return CHOOSING

def chatgpt_send(userPrompt):
    max_tokens = 500
    prompt_content = f"{userPrompt}, response length {max_tokens} characters."
    
    prompt = [
        {
            "role": "user", 
            "content": prompt_content
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
      # if you don't have gpt-4 access, use 
      # model="gpt-3.5-turbo"
        messages=prompt,
        max_tokens=max_tokens,
        n=1,
        temperature=0
    )
    
    chatgpt_response = response['choices'][0]['message']['content']
    
    return chatgpt_response

def main() -> None:
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.regex('^PUSH ME AND THEN...$'), want_start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, user_prompt))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

