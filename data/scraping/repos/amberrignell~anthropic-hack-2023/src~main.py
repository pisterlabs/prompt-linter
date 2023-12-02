import logging
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, InlineQueryHandler

from prompts import create_conversation_prompt, format_conversation_response

load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

anthropic = Anthropic()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_caps = ' '.join(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

chat_histories = {}

async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message.text

    prompt = create_conversation_prompt(message, chat_id)
    print(prompt)

    # Call Claude's API with the prompt
    completion = anthropic.completions.create(
        model="claude-2.0",
        max_tokens_to_sample=350,
        temperature=1,
        prompt=prompt,
    )
    
    raw_response = completion.completion
    print(raw_response)

    formatted_response = format_conversation_response(raw_response)
    print("chatbot message \n", formatted_response)

    # Send Claude's response to the Telegram chat
    await context.bot.send_message(parse_mode="HTML", chat_id=update.effective_chat.id, text=formatted_response)

async def inline_caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    results = []
    results.append(
        InlineQueryResultArticle(
            id=query.upper(),
            title='Caps',
            input_message_content=InputTextMessageContent(query.upper())
        )
    )
    await context.bot.answer_inline_query(update.inline_query.id, results)

if __name__ == '__main__':
    application = ApplicationBuilder().token(AUTH_TOKEN).build()

    start_handler = CommandHandler('start', start)
    # echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    response_handler = MessageHandler(
        filters.TEXT & (~filters.COMMAND), respond)
    caps_handler = CommandHandler('caps', caps)
    # news_handler = CommandHandler('news', news)
    inline_caps_handler = InlineQueryHandler(inline_caps)

    application.add_handler(start_handler)
    application.add_handler(response_handler)
    application.add_handler(caps_handler)
    # application.add_handler(news_handler)
    application.add_handler(inline_caps_handler)

    application.run_polling()
