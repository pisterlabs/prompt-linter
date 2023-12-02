import os
from telethon import TelegramClient, events, types
from telethon.sessions import StringSession
import openai
import json
import datetime
import re
import configparser
import traceback
import aiohttp
import time

import src.db_helper as db_helper
import src.openai_helper as openai_helper
import src.announce_helper as announce_helper
import src.logging_helper as logging
import src.userdailyactivity_helper as userdailyactivity_helper

config = configparser.ConfigParser(os.environ)
config_path = os.path.dirname(__file__) + '/../config/' #we need this trick to get path to config folder
config.read(config_path + 'settings.ini')

logger = logging.get_logger()

#TODO:HIGH: move env variables to .env file
# Get API credentials from environment variables
#TODO:MED: rewrite this with logging module

openai.api_key = config['OPENAI']['KEY']
client = TelegramClient(StringSession(config['TELEGRAM']['SESSION_STRING']), config['TELEGRAM']['API_ID'], config['TELEGRAM']['API_HASH'])

async def safe_send_message(chat_id, message, link_preview=False):
    try:
        #split message into chunks of 4096 chars
        message_chunks = [message[i:i + 4096] for i in range(0, len(message), 4096)]
        for message_chunk in message_chunks:
            await client.send_message(chat_id, message_chunk, link_preview=link_preview)
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await client.send_message(chat_id, f"Error: {e}. Please try again later.")

async def safe_send_image(chat_id, image):
    try:
        await client.send_file(chat_id, image)
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await client.send_message(chat_id, f"Error: {e}. Please try again later.")

async def get_last_x_messages(client, channel_id, max_tokens = 4000):
    me = await client.get_me()

    channel = await client.get_entity(channel_id)

    # Fetch messages until '/clear' or until max tokens
    messages = []
    total_tokens = 0
    min_id = None

    async for msg in client.iter_messages(channel):

        #TODO:MED: maybe we need to take not whole coversation, but only last day. So we don't use context from old messages
        if msg.text == '/clear' or msg.text == 'Conversation history cleared':
            break

        if msg.text is None:
            continue

        if total_tokens + len(msg.text) <= max_tokens:
            if msg.sender == me:
                messages.append({"role": "assistant", "content": msg.text})
            else:
                messages.append({"role": "user", "content": msg.text})

            total_tokens += len(msg.text)
            min_id = msg.id
        else:
            break

    return messages[::-1]

async def handle_image_command(event, session):
    try:
        if not event.text.startswith('/image'):
            return
        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/image', usage_count=1)

        user = session.query(db_helper.User).filter_by(id=event.chat_id).first()

        prompt = event.text[len('/image'):].strip()

        if prompt:
            await safe_send_message(event.chat_id, "Generating images...\n(can take 1-2 minutes)")

            try:
                images_url = openai_helper.generate_image(prompt)
                if images_url is not None:
                    for image_url in images_url:
                        async with aiohttp.ClientSession() as httpsession:
                            async with httpsession.get(image_url['url']) as response:
                                await safe_send_image(event.chat_id, await response.read())
                else:
                    await safe_send_message(event.chat_id, "Error: Failed to generate image with this text. Please try again later.")
            except Exception as e:
                logger.error(f"Error: {traceback.format_exc()}")
                await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")
        else:
            await safe_send_message(event.chat_id, "Please provide a prompt for the image")
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_remember_command(event, session):
    try:
        if not event.text.startswith('/remember'):
            return

        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/remember', usage_count=1)

        user = session.query(db_helper.User).filter_by(id=event.chat_id).first()

        memory_text = event.text[len('/remember'):].strip()

        if memory_text:
            user.memory = memory_text
            await safe_send_message(event.chat_id, f"Memory has been set to: '{memory_text}'")
        else:
            user.memory = ''
            await safe_send_message(event.chat_id, "Memory has been cleared")

        session.commit()
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_memory_command(event, session):
    try:
        if not event.text.startswith('/memory'):
            return

        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/memory', usage_count=1)

        user = session.query(db_helper.User).filter_by(id=event.chat_id).first()

        if user.memory:
            await safe_send_message(event.chat_id, f"Current memory: '{user.memory}'")
        else:
            await safe_send_message(event.chat_id, "Memory is not set. If you'd like to set a memory, you can do so by typing /remember followed by the text you'd like to use as the memory.")
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_start_command(event):
    try:
        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/start', usage_count=1)

        welcome_text = """
    Hi! I'm a bot that uses OpenAI's GPT-4 to talk to you.
    Just write me a message and I'll try to respond to it or use one of the following commands.
    
    Commands:
    /image [TEXT]        - generate a set of images based on the text.
    /remember [TEXT]     - set a memory that will be used in the conversation.
    /memory              - show the current memory.
    /clear               - clear the conversation history (don't use previous messages to generate a response).
    /help                - show this message.
    /s or /summary       - get summary of the text or url. E.g. /summary https://openai.com/product/gpt-4
    
    Don't forget to subscribe to ❗️@rvnikita_blog ❗ - Nikita Rvachev's blog. if you want to get more updates about GPT-4, AI and this bot.
            """

        await safe_send_message(event.chat_id, welcome_text)
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_default(event):
    try:
        #TODO:HIGH add url extraction (think how to work if text size is bigger then prompt limit)
        if event.text.startswith('/'):
            userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command="/unknown", usage_count=1)

            await safe_send_message(event.chat_id, "Unknown command")
            await handle_start_command(event)
            return

        with db_helper.session_scope() as session:
            user = session.query(db_helper.User).filter_by(id=event.chat_id).first()

            async with client.action(event.chat_id, 'typing'):
                conversation_history = await get_last_x_messages(client, event.chat_id, 4000)
                response, prompt_tokens, completion_tokens, google_used = await openai_helper.generate_response(conversation_history,
                                                                                                   user.memory,
                                                                                                   model=user.openai_model)

                userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command=None,
                                                                  prompt_tokens=prompt_tokens,
                                                                  completion_tokens=completion_tokens, usage_count=1)

                if google_used == True:
                    userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command="/google", usage_count=1)

                session.commit()

        await safe_send_message(event.chat_id, response)
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_summary_command(event):
    try:
        if event.text.startswith('/summary'):
            url_or_text = event.text[len('/summary'):].strip()
        elif event.text.startswith('/s '):
            url_or_text = event.text[len('/s '):].strip()
        else:
            return

        #TODO:HIGH: we need to calculate tokens here with special library
        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/summary', usage_count=1)

        # check if it's a url_or_text is empty (only spaces,tabs or nothing)
        if re.match(r"^[\s\t]*$", url_or_text):
            await safe_send_message(event.chat_id, "You need to provide an url or text after /summary get summary. E.g. /summary https://openai.com/product/gpt-4")
            return

        if url_or_text is None:
            await safe_send_message(event.chat_id, "You need to provide an url or text after /summary get summary. E.g. /summary https://openai.com/product/gpt-4")
            return

        await safe_send_message(event.chat_id, "Generating summary...\n(can take 2-3 minutes for big pages)")

        async with client.action(event.chat_id, 'typing', delay=5):

            try:
                url_content_title, url_content_body = openai_helper.get_url_content(url_or_text)
            except Exception as e:
                logger.error(f"Error: {traceback.format_exc()}")
                await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")
                return
            with db_helper.session_scope() as session:
                user = session.query(db_helper.User).filter_by(id=event.chat_id).first()

                # check if it's a url or a text
                if url_content_body is not None:  # so that was a valid url
                    summary, prompt_tokens, completion_tokens = openai_helper.get_summary_from_text(url_content_body, url_content_title, model=user.openai_model)
                else:  # so that was a text
                    # FIXME: we can get url_content_body = None even for valid url. So this else is not 100% correct
                    summary, prompt_tokens, completion_tokens = openai_helper.get_summary_from_text(url_or_text, model=user.openai_model)

                if prompt_tokens != 0:
                    userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/summary', prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

                await safe_send_message(event.chat_id, summary)
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_test_announcement_command(event, session):
    try:
        if not event.text.startswith('/test_announcement'):
            return

        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/test_announcement', usage_count=1)

        if event.sender_id != int(config['TELEGRAM']['ADMIN_ID']):
            return

        announcement_text = event.text[len('/test_announcement'):].strip()
        if announcement_text:
            await announce_helper.add_message_to_queue(announcement_text, is_test=True, session=session)
            await safe_send_message(event.chat_id, "Test announcement added to queue")
        else:
            await safe_send_message(event.chat_id, "Please provide a text after /test_announcement. E.g. /test_announcement Hello, this is a test announcement!")
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def handle_announcement_command(event, session):
    try:
        if not event.text.startswith('/announcement'):
            return

        userdailyactivity_helper.update_userdailyactivity(user_id=event.chat_id, command='/announcement', usage_count=1)

        #could be used only by admins
        if event.sender_id != int(config['TELEGRAM']['ADMIN_ID']):
            return

        announcement_text = event.text[len('/announcement'):].strip()
        if announcement_text:
            await announce_helper.add_message_to_queue(announcement_text, is_test=False, session=session)
            await safe_send_message(event.chat_id, "Announcement added to queue")
        else:
            await safe_send_message(event.chat_id, "Please provide a text after /announcement. E.g. /announcement Hello, this is an announcement!")
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")

async def on_new_message(event):
    #TODO:HIGH protect from spam attack from users (if they send to much messages in a short period of time)
    try:
        with db_helper.session_scope() as session:
            if event.is_private != True:
                return
            if event.sender_id == (await client.get_me()).id:
                return

            # Add this to get event.chat_id entity if this is first time we see it
            try:
                user_info = await client.get_entity(event.chat_id)
            except:
                await client.get_dialogs()
                user_info = await client.get_entity(event.chat_id)

            user = session.query(db_helper.User).filter_by(id=event.chat_id).first()

            if user is None:
                user = db_helper.User(id=event.chat_id, status='active', memory='', username=user_info.username, first_name=user_info.first_name, last_name=user_info.last_name, last_message_datetime=datetime.datetime.now())
                session.add(user)
                session.commit()

                await handle_start_command(event)
                await handle_default(event)
                return
            else:
                if user.username is None:
                    user.username = user_info.username
                if user.first_name is None:
                    user.first_name = user_info.first_name
                if user.last_name is None:
                    user.last_name = user_info.last_name
                user.last_message_datetime = datetime.datetime.now()
                session.commit()

            user_daily_activity = session.query(db_helper.UserDailyActivity).filter_by(user_id=event.chat_id,date=datetime.date.today()).first()
            if user_daily_activity is None:
                user_daily_activity = db_helper.UserDailyActivity(user_id=user.id, date=datetime.date.today())
                session.add(user_daily_activity)

            if event.text.startswith('/test_announcement'):
                await handle_test_announcement_command(event, session=session)
                return

            if event.text.startswith('/announcement'):
                await handle_announcement_command(event, session=session)
                return

            if event.text == '/clear':
                await safe_send_message(event.chat_id, "Conversation history cleared")
                return

            if event.text == '/start' or event.text == '/help':
                await handle_start_command(event)
                return

            if event.text.startswith('/remember'):
                await handle_remember_command(event, session=session)
                return

            if event.text.startswith('/memory'):
                await handle_memory_command(event, session=session)
                return

            if event.text.startswith('/summary') or event.text.startswith('/s '):
                await handle_summary_command(event)
                return

            if event.text.startswith('/image'):
                await handle_image_command(event, session=session)
                return

            await handle_default(event)
            return

    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        await safe_send_message(event.chat_id, f"Error: {e}. Please try again later.")


async def main():
    # Initialize the Telegram client
    # Connect and sign in using the phone number

    logger.info("Connecting to Telegram...")

    backoff = 1
    while True:
        try:
            await client.start()
            client.add_event_handler(on_new_message, events.NewMessage)
            await client.run_until_disconnected()
            break
        except Exception as e:
            logger.error(f"Error: {traceback.format_exc()}")
            logger.error("Session string was used from two different IPs. Regenerating new session string.")
            logger.info(f"Retry connecting in {backoff} seconds...")

            # Use this time to regenerate a new session string if necessary
            # Remember to replace "regenerate_session_string()" with your actual function
            # session_string = regenerate_session_string()
            # client = TelegramClient(StringSession(session_string), API_ID, API_HASH)

            time.sleep(backoff)  # Wait for the specified backoff time
            backoff *= 2  # Double the backoff time for the next retry

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())