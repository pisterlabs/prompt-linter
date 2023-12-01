from telegram.ext import CallbackContext, CommandHandler, MessageHandler, Filters
from telegram import Update, ChatAction
import os
#from chatgpt.api import ChatGPT
import openai as oa
import random

enabled = True

ctx = None
blocked = False
blocked_show = 0
ALLOWED_CONDITION = (
    Filters.chat_type.group &
    Filters.text &
    (~ Filters.forwarded) &
    (
        Filters.chat(-1001171487755) |
        Filters.chat(-511871661)
    )
)
ADMIN_LENGTH = 768
USER_LENGTH = 384


def load():
    global ctx
    # ctx = ChatGPT(session_token=os.getenv(
    #    "MODULE_CHAT_TOKEN"), response_timeout=30, user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15")
    oa.api_key = os.getenv("MODULE_CHAT_KEY")
    print("Chat Plugin Loaded!")


def exec_chat(msg, length=384):
    # return ctx.send_message(msg).content
    return oa.Completion.create(model="text-davinci-003", prompt=msg, temperature=0, max_tokens=length, request_timeout=40).choices[0].text


def chat_cmd(update: Update, context: CallbackContext) -> None:
    length = USER_LENGTH
    global blocked, blocked_show
    if len(context.args) == 0:
        update.message.reply_text(
            "*Chat with bot using ChatGPT.*\nUsage: `/chat {msg}`.", parse_mode='Markdown')
    elif not blocked:
        if str(update.message.from_user.id) == str(os.getenv("MODULE_CHAT_ADMIN")):
            length = ADMIN_LENGTH
        blocked = True
        context.bot.sendChatAction(
            chat_id=update.message.chat_id, action=ChatAction.TYPING)
        try:
            result = exec_chat(" ".join(context.args), length=length)
            update.message.reply_text(result, parse_mode='Markdown')
        except Exception as e:
            if str(e):
                update.message.reply_text(
                    "`Error: {}`".format(str(e)), parse_mode='Markdown')
            else:
                update.message.reply_text(
                    "`Error: Network`", parse_mode='Markdown')
        finally:
            blocked = False
            blocked_show = 0
    elif blocked_show < 2:
        update.message.reply_text(
            "`Error: Busy`", parse_mode='Markdown')
        blocked_show = blocked_show + 1


def chat_random(update: Update, context: CallbackContext) -> None:
    global blocked, blocked_show
    if blocked:
        return
    if (len(update.message.text) <= 10):
        return
    if (random.randint(0, 256) < 245):
        return
    blocked = True
    context.bot.sendChatAction(
        chat_id=update.message.chat_id, action=ChatAction.TYPING)
    try:
        result = exec_chat(update.message.text)
        update.message.reply_text(result, parse_mode='Markdown')
    except:
        pass
    finally:
        blocked = False
        blocked_show = 0


handlers = [CommandHandler("chat", chat_cmd, run_async=True),
            MessageHandler(ALLOWED_CONDITION, chat_random)]

if __name__ == "__main__":
    pass
