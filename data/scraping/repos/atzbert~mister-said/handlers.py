import asyncio

from google.cloud import firestore
from telegram import Update, ChatMember, Chat
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from functools import wraps

import config
import helpers
from config import TELEGRAM_BOT, MAXIMUM_CHATS
from helpers import increment_message_count, translate_and_send_messages, increment_active_chats
from openai_helper import get_openai_response
from telegram.error import TelegramError

db = firestore.Client(TELEGRAM_BOT)


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        async def command_func(update, context, *args, **kwargs):
            await context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return await func(update, context,  *args, **kwargs)
        return command_func

    return decorator


send_typing_action = send_action(ChatAction.TYPING)

@send_typing_action
async def greet_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    new_users = update.effective_message.new_chat_members
    print(f"New users {new_users} added to  chat {chat_id}")

    # Check if the bot is added to the chat
    bot_added = False
    for user in new_users:
        if user.is_bot and user.username == context.bot.username:
            bot_added = True

    # If the bot is added, check the active chat count
    if bot_added and not await increment_active_chats():
        await context.bot.send_message(chat_id=chat_id,
                                       text=f"Sorry, I can't join this chat. I'm already in ${MAXIMUM_CHATS} chats.")
        await context.bot.leave_chat(chat_id=chat_id)
        return

    # If the bot is not added, or the active chat count is below the limit, greet new users
    for user in new_users:
        try:
            await context.bot.send_message(chat_id=chat_id,
                                           text=f"Welcome to the chat, {user.full_name}! I'm Mister Said, a bot that can automatically translate messages in group chats. "
                                                "Please use the '/setlang [code]' command to set your preferred language. "
                                                "Please use only supported language codes which you can find here: https://cloud.google.com/translate/docs/languages")
        except TelegramError as e:
            print(f"Error sending greeting message: {e}")
            continue

@send_typing_action
async def translate_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    message_text = update.effective_message.text

    bot_mention = f"@{context.bot.username}"
    # Get the number of members in the chat
    chat_info: Chat = await context.bot.get_chat(chat_id)
    chat_member_count = await chat_info.get_member_count()

    if chat_member_count == 2 or message_text.startswith(bot_mention):
        if message_text.startswith(bot_mention):
            message_text = message_text[len(bot_mention):].strip()
        history = await helpers.get_previous_messages(chat_id, user_id)
        msg = await helpers.store_message(chat_id, user_id, message_text)
        if len(history) == 0:
            history.extend(helpers.init_messages)
            for init_message in helpers.init_messages:
                await helpers.store_message(chat_id, user_id, init_message['content'], role=init_message['role'])
        history.append(msg)
        openai_response = await get_openai_response(history)
        if openai_response:
            await helpers.store_message(chat_id=chat_id,user_id=user_id, role="assistant", message_text=openai_response)
            try:
                await context.bot.send_message(chat_id=chat_id, text=openai_response)
            except TelegramError as e:
                print(f"Error sending OpenAI response: {e}")
        return

    if await increment_message_count(chat_id) > config.MESSAGE_LIMIT:
        print(f"Message limit exceeded in chat {chat_id}")
        return

    await translate_and_send_messages(update, context, message_text)


async def remove_left_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    left_user = update.effective_message.left_chat_member
    user_id = left_user.id

    doc_ref = db.collection(u'chats').document(str(chat_id)).collection(u'members').document(str(user_id))
    doc_ref.delete()
    print(f"Removed language preferences for user {user_id} in chat {chat_id}")


async def bot_removed_from_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    my_chat_member = update.my_chat_member
    print(f"Bot modified in chat {chat_id}: {my_chat_member.difference()}")
    if my_chat_member.old_chat_member.status == ChatMember.ADMINISTRATOR or my_chat_member.old_chat_member.status == ChatMember.MEMBER:
        if my_chat_member.new_chat_member.status == ChatMember.BANNED or my_chat_member.new_chat_member.status == ChatMember.LEFT:
            doc_ref = db.collection(u'chats').document(str(chat_id))
            doc_ref.delete()
            print(f"Removed chat {chat_id} from the database.")


async def bot_added_to_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    print(f"Bot modified in chat {chat_id}")
    my_chat_member = update.my_chat_member
    if (my_chat_member.old_chat_member == None and
        my_chat_member.new_chat_member.status == ChatMember.ADMINISTRATOR) or \
            my_chat_member.new_chat_member.status == ChatMember.MEMBER:
        doc_ref = db.collection(u'chats').document(str(chat_id))
        doc_ref.create({'title': update.effective_chat.title})
        print(f"Bot added to chat {chat_id}")
