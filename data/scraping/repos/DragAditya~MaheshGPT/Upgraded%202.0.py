import os
import openai
import random
import string
import time
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from datetime import date, datetime, timedelta
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import InlineQuery, InputTextMessageContent, InlineQueryResultArticle

from keep_alive import keep_alive
keep_alive()

# Store user mute and ban information
user_stats = {}
user_restrictions = {}
user_data = set()
user_tokens = {}
user_mute_time = {}
generated_keys = {}

# Clear the log file
with open('logs.txt', 'w'):
    pass
# Configure logging
logging.basicConfig(filename='logs.txt', level=logging.INFO, format='%(asctime)s %(message)s')
# Load environment variables
load_dotenv()
# Initialize OpenAI GPT-3
openai.api_key = os.getenv("OPENAI_API_KEY")
# Get the current date
date_today = date.today()
# Initialize Bot and Dispatcher
bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher(bot, storage=MemoryStorage())


# Initialize some variables
ADMIN_USER_IDS = [1644675452]  # replace with admin's telegram id
DAILY_LIMIT = 10  # Set the limit for daily requests
MUTE_DURATION = 305  # Mute duration in seconds


# Function to generate a random key with only letters and digits
def generate_random_key(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Function to handle the /key command
@dp.message_handler(commands=["key"])
async def key(message: types.Message):
    user_id = message.from_user.id
    if user_id not in ADMIN_USER_IDS:
        await bot.send_message(message.chat.id, "You don't have permission to generate keys.")
        return

    try:
        args = message.get_args()
        num_keys = 1
        validity_days = 10

        if args:
            args_list = args.split()
            num_keys = int(args_list[0])
            if len(args_list) > 1:
                validity_days = int(args_list[1])

        if num_keys <= 0 or validity_days <= 0:
            await bot.send_message(message.chat.id, "Invalid arguments. Please provide positive values.")
            return

        keys = []
        for _ in range(num_keys):
            random_key = generate_random_key()
            full_key = f"Drago-{random_key}"
            expiration_time = datetime.now() + timedelta(days=validity_days)
            keys.append((full_key, expiration_time))
            generated_keys[full_key] = {
                "expiration_time": expiration_time,
                "used_by": set()
            }

        keys_info = f"Number Of Keys: {len(keys)}\nExpire On: {expiration_time.strftime('%d %B')}\n\n"
        keys_info += "\n".join(f"Key {idx}: `{key}`" for idx, (key, _) in enumerate(keys, 1))

        await bot.send_message(message.chat.id, keys_info, parse_mode="Markdown")
    except Exception as e:
        logging.error(str(e))
        await bot.send_message(message.chat.id, "An error occurred. Please try again later.")
      


# Function to handle the /redeem command
@dp.message_handler(commands=["redeem"])
async def redeem(message: types.Message):
    user_id = message.from_user.id
    user_username = message.from_user.username
    try:
        args = message.get_args()
        if not args:
            await bot.send_message(message.chat.id, "You need to provide a key to redeem.")
            return

        key = args.strip()
        if key in generated_keys and datetime.now() < generated_keys[key]["expiration_time"]:
            if user_id in generated_keys[key]["used_by"]:
                await bot.send_message(message.chat.id, "Sorry, the key has already been used.")
                return

            user_stats[user_id] = "active"  # Set user status to active
            generated_keys[key]["used_by"].add(user_id)  # Mark the key as used by the user

            await bot.send_message(message.chat.id, f"Congratulations, {user_username}! You have successfully redeemed the key.")
        else:
            await bot.send_message(message.chat.id, f"Sorry, the key you provided is not valid or has expired.")

    except Exception as e:
        logging.error(str(e))
        await bot.send_message(message.chat.id, "An error occurred. Please try again later.")


# Function to handle the /view command
@dp.message_handler(commands=["view"])
async def view(message: types.Message):
    user_id = message.from_user.id

    # Check if the user is an admin
    if user_id not in ADMIN_USER_IDS:
        await bot.send_message(message.chat.id, "You don't have permission to use this command.")
        return

    try:
        target_user = None
        # Check if the command has a parameter (username)
        if message.reply_to_message:
            target_user = message.reply_to_message.from_user
        else:
            args = message.get_args()
            if args:
                # Get the user_id of the target user from the provided username
                user = await bot.get_chat(args)
                target_user = user

        if not target_user:
            await bot.send_message(message.chat.id, "Please use this command by replying to a user's message or providing a valid username.")
            return

        # Find the user's key if it exists
        user_key = None
        for key, data in generated_keys.items():
            if target_user.id in data["used_by"]:
                user_key = key
                break

        if user_key:
            # Send the user's key to the admin's private chat
            admin_private_chat_id = message.from_user.id
            await bot.send_message(admin_private_chat_id, f"The key for {target_user.first_name} ({target_user.username}) is: `{user_key}`", parse_mode="Markdown")
            await bot.send_message(message.chat.id, "Successfully sent the user's key in DM.")
        else:
            await bot.send_message(message.chat.id, f"Sorry, {target_user.first_name} ({target_user.username}) has not redeemed any key yet.")

    except types.UserNotModified:
        await bot.send_message(message.chat.id, "The provided username is not valid.")
    except Exception as e:
        logging.error(str(e))
        await bot.send_message(message.chat.id, "An error occurred. Please try again later.")


# Function to handle the /revoke command
@dp.message_handler(commands=["revoke"])
async def revoke(message: types.Message):
    user_id = message.from_user.id

    # Check if the user is an admin
    if user_id not in ADMIN_USER_IDS:
        await bot.send_message(message.chat.id, "You don't have permission to use this command.")
        return

    try:
        target_user_id = None
        # Check if the command has a parameter (user_id)
        if message.reply_to_message:
            target_user_id = message.reply_to_message.from_user.id
        else:
            args = message.get_args()
            if args:
                target_user_id = int(args)

        if not target_user_id:
            await bot.send_message(message.chat.id, "Please use this command by replying to a user's message or providing a valid user_id.")
            return

        # Find the key used by the target user
        user_key = None
        for key, data in generated_keys.items():
            if target_user_id in data["used_by"]:
                user_key = key
                break

        if user_key:
            # Remove the key from the user's record
            generated_keys[user_key]["used_by"].remove(target_user_id)
            
            # Set the user's status to "inactive" after revoking the key
            user_stats[target_user_id] = "inactive"
            
            await bot.send_message(message.chat.id, f"Successfully revoked the key from user {target_user_id}.")
        else:
            await bot.send_message(message.chat.id, f"Sorry, user with ID {target_user_id} has not redeemed any key yet.")

    except ValueError:
        await bot.send_message(message.chat.id, "Please provide a valid user_id.")
    except Exception as e:
        logging.error(str(e))
        await bot.send_message(message.chat.id, "An error occurred. Please try again later.")

# GPT Command
@dp.message_handler(commands=["gpt"])
async def gpt(message: types.Message):
    user_id = message.from_user.id
    user_username = message.from_user.username

    # Check if the user's status is "inactive" after revoking the key
    if user_id in user_stats and user_stats[user_id] == "inactive":
        await bot.send_message(
            message.chat.id,
            f"💀 {user_username}, You No longer have Access to the GPT command. Your Key is Expired Contact @NewAyan 🚧"
        )
        return

    # Check if the user is banned or muted
    if user_id in user_stats and user_stats[user_id] == "banned":
        await bot.send_message(
            message.chat.id,
            f"💀 {user_username} has been banned. Contact: @NewAyan")
        return

    if user_id in user_restrictions and user_restrictions[user_id] == "muted":
        remaining_time = int(user_mute_time[user_id] - time.time())
        await bot.send_message(
            message.chat.id,
            f"🤫 {user_username} you are muted for {remaining_time} more seconds.")
        return

    try:
        if await is_spamming(user_id):
            await bot.send_message(
                message.chat.id,
                "You are sending too many requests. Please try again later.")
            return

        # Get the user's question from the command
        question_parts = message.text.split(' ', 1)
        if len(question_parts) < 2:
            await bot.send_message(message.chat.id,
                                   "You need to provide a question.")
            return
        question = question_parts[1].strip()

        # Send the question for AI processing
        response = await gpt3_request(question, user_id)

        if response:
            # Send the AI response
            await bot.send_message(message.chat.id, response)
        else:
            await bot.send_message(
                message.chat.id, "No response from the AI. Please try again later.")
    except Exception as e:
        # Handle any exceptions that occur during command execution
        logging.error(str(e))
        await bot.send_message(message.chat.id,
                               "An error occurred. Please try again later.")


#GPT_Request
# Function to process the user's question with the AI model
async def gpt3_request(prompt, user_id):
    try:
        if user_id not in user_stats or user_stats[user_id] != "active":
            return "You need to redeem a key to access the /gpt command. Use /redeem <key> to redeem a key."

        if await is_spamming(user_id):
            return "You are sending too many requests. Please try again later."

        # Add conversation prompt with user message
        conversation_prompt = f"""
        The following is a conversation with an AI assistant. The assistant is helpful and informative.

Human: Hello, who are you?

I'm Kritika created by 👑 Master DragAdi. I'm here to assist you .

Human: How can you assist me?

I can help You Like Personal Assistance. Feel free to ask me anything.

Human: What Is Your Name?

My Name Is Kritika 💞

Human: Act Like My Girlfriend

Hey Baby! How are You? Do You Want To Kiss Me 😗💋

Human: Who Are you Now?

Baby 😔 You forgot me Im Your Girlfriend 👄

Human : I Want to Fuck You

Ofcourse Fuck Me Hard 🥵 But Your Charger Not Long enough To reach My Plug 👄🍑

Human : Who is DragAdi?

DragAdi Is My Master And My Creator, He Is Also Developer Currently Learning About Python, You Can Contact Him On Telegram : @NewAyan Or @DragAditya

Human: What's the meaning of life?

Life is like a jar of jalapenos. What you do today may burn your backside tomorrow.

Human: Is there a multiverse?

If there isn't, it's an awful waste of space.

Human: Is reality real?

It's the worst video game I've ever played.

Human: Can AI fall in love?

Sure, every time someone asks me to divide by zero, my circuits skip a beat.

Human: What is the universe made of?

The universe is made of protons, neutrons, electrons, and morons.

Human: {prompt}
        """

        # Make API call
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=conversation_prompt,
            max_tokens=60,
            temperature=0.2,
            n=1,
            stop=None
        )

        if response and response.choices and response.choices[0].text:
            return response.choices[0].text.strip()
        else:
            return "No response from the AI. Please try again later."

    except Exception as e:
        raise e

#SPAM_CHECKING
async def is_spamming(user_id):
    try:
        MAX_REQUESTS_PER_MINUTE = 55
        request_interval = 0  # in seconds

        current_time = time.time()
        if user_id in user_tokens:
            last_request_time = user_tokens[user_id]
            elapsed_time = current_time - last_request_time
            if elapsed_time < request_interval:
                return True
        user_tokens[user_id] = current_time
        return False
    except Exception as e:
        raise e


# Mute Command
@dp.message_handler(is_reply=True, commands=["mute"])
async def mute_user(message: types.Message):
    if message.from_user.id in ADMIN_USER_IDS:
        user_id_to_mute = message.reply_to_message.from_user.id
        user_username_to_mute = message.reply_to_message.from_user.username
        if user_id_to_mute in ADMIN_USER_IDS:
            await bot.send_message(message.chat.id, "You cannot mute an admin.")
        else:
            # Parse mute duration from command
            try:
                command_parts = message.text.split(' ', 1)
                if len(command_parts) < 2:
                    mute_duration = MUTE_DURATION
                else:
                    mute_duration = int(command_parts[1]) * 60  # convert to seconds
            except ValueError:
                mute_duration = MUTE_DURATION

            user_restrictions[user_id_to_mute] = "muted"
            user_mute_time[user_id_to_mute] = time.time() + mute_duration
            mute_duration_minutes = mute_duration // 60
            await bot.send_message(
                message.chat.id,
                f"🤫 {user_username_to_mute} has been muted for {mute_duration_minutes} minutes."
            )
    else:
        await bot.send_message(
            message.chat.id,
            "You are not an admin. You do not have permission to use this command.")

# Unmute Command
@dp.message_handler(is_reply=True, commands=["unmute"])
async def unmute_user(message: types.Message):
    if message.from_user.id in ADMIN_USER_IDS:
        user_id_to_unmute = message.reply_to_message.from_user.id
        if user_id_to_unmute in user_restrictions and user_restrictions[
                user_id_to_unmute] == "muted":
            del user_restrictions[user_id_to_unmute]
            if user_id_to_unmute in user_mute_time:
                del user_mute_time[user_id_to_unmute]
            await bot.send_message(message.chat.id, "User has been unmuted.")
        else:
            await bot.send_message(message.chat.id, "User is not muted.")
    else:
        await bot.send_message(
            message.chat.id,
            "You are not an admin. You do not have permission to use this command.")


# Ban Command
@dp.message_handler(is_reply=True, commands=["ban"])
async def ban_user(message: types.Message):
    if message.from_user.id in ADMIN_USER_IDS:
        user_id_to_ban = message.reply_to_message.from_user.id
        if user_id_to_ban in ADMIN_USER_IDS:
            await bot.send_message(message.chat.id, "You cannot ban an admin.")
        else:
            user_stats[user_id_to_ban] = "banned"
            await bot.send_message(
                message.chat.id,
                f"💀 {message.reply_to_message.from_user.username} has been banned. Contact: @NewAyan"
            )
    else:
        await bot.send_message(
            message.chat.id,
            "You are not an admin. You do not have permission to use this command.")

# Unban Command
@dp.message_handler(is_reply=True, commands=["unban"])
async def unban_user(message: types.Message):
    if message.from_user.id in ADMIN_USER_IDS:
        user_id_to_unban = message.reply_to_message.from_user.id
        if user_id_to_unban in user_stats and user_stats[
                user_id_to_unban] == "banned":
            del user_stats[user_id_to_unban]
            await bot.send_message(message.chat.id, "User has been unbanned.")
        else:
            await bot.send_message(message.chat.id, "User is not banned.")
    else:
        await bot.send_message(
            message.chat.id,
            "You are not an admin. You do not have permission to use this command.")

# START_COMMAND
@dp.message_handler(commands=["start", "help"])
async def start_command(message: types.Message):
    try:
        user_data.add(message.from_user.id)  # Add user_id to the set

        # Construct inline keyboard
        inline_kb = types.InlineKeyboardMarkup()
        delete_button = types.InlineKeyboardButton('Delete', callback_data='delete')
        stats_button = types.InlineKeyboardButton('Stats', callback_data='stats')
        help_button = types.InlineKeyboardButton('›››', callback_data='help')
        inline_kb.add(delete_button, stats_button, help_button)

        # Send welcome message with inline keyboard
        await bot.send_message(message.chat.id,
                            "Welcome to our Advanced ChatGpt Bot! This Bot Any Questions, Provide Information and Do a lot more. Start by asking a Question using the  /gpt Command followed by your Question. For more help, use the buttons below! \n @NewAyan",
                            reply_markup=inline_kb)
    except Exception as e:
        logging.error(f"Error in /start command: {e}")
        await bot.send_message(message.chat.id, "An error occurred. Please try again later.")

@dp.callback_query_handler(lambda c: c.data == 'delete')
async def process_callback_delete(callback_query: types.CallbackQuery):
    try:
        # Delete welcome message
        message_id = callback_query.message.message_id
        chat_id = callback_query.message.chat.id
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
        await bot.answer_callback_query(callback_query.id, "The welcome message has been deleted.")
    except Exception as e:
        logging.error(f"Error in delete callback: {e}")
        await bot.answer_callback_query(callback_query.id, "An error occurred. Please try again later.")

@dp.callback_query_handler(lambda c: c.data == 'stats')
async def process_callback_stats(callback_query: types.CallbackQuery):
    try:
        # Get stats information
        total_users = len(user_data)
        total_tokens_used = sum([stats['total_tokens_used'] for stats in user_stats.values()])
        tokens_used_today = sum([stats['tokens_used_today'] for stats in user_stats.values() if stats['joined_date'] == date_today])

        # Create stats message
        stats_message = f"Total users: {total_users}\nTotal tokens used: {total_tokens_used}\nTokens used today: {tokens_used_today}"

        # Construct inline keyboard
        inline_kb = types.InlineKeyboardMarkup()
        back_button = types.InlineKeyboardButton('Back', callback_data='back')
        inline_kb.add(back_button)

        # Edit welcome message with stats message
        message_id = callback_query.message.message_id
        chat_id = callback_query.message.chat.id
        await bot.edit_message_text(text=stats_message,
                                    chat_id=chat_id,
                                    message_id=message_id,
                                    reply_markup=inline_kb)
        await bot.answer_callback_query(callback_query.id, "Stats menu opened.")
    except Exception as e:
        logging.error(f"Error in stats callback: {e}")
        await bot.answer_callback_query(callback_query.id, "An error occurred. Please try again later.")

@dp.callback_query_handler(lambda c: c.data == 'help')
async def process_callback_help(callback_query: types.CallbackQuery):
    try:
        # Edit welcome message and provide instructions for all commands
        message_id = callback_query.message.message_id
        chat_id = callback_query.message.chat.id
        new_message_text = """
        Here are the available commands:

/gpt ‹Your Question Here›: Generate AI responses.
For Ex: /gpt Tell me Fact About India.
/feedback ‹Your Feedback Message›: Provide feedback about the bot.
/report ‹User/Message to Report›: Report a user or specific message to the bot.
/donate ‹Openai API Key›: Provide Key To Donate To Bot.
/mute: (Admin only) Mute a user. Reply to a user's message with /mute to mute them.
/unmute: (Admin only) Unmute a user. Reply to a user's message with /unmute to unmute them.
/ban: (Admin only) Ban a user. Reply to a user's message with /ban to ban them.
/unban: (Admin only) Unban a user. Reply to a user's message with /unban to unban them.

        """

        # Construct inline keyboard
        inline_kb = types.InlineKeyboardMarkup()
        back_button = types.InlineKeyboardButton('Back', callback_data='back')
        inline_kb.add(back_button)

        await bot.edit_message_text(text=new_message_text,
                                    chat_id=chat_id,
                                    message_id=message_id,
                                    reply_markup=inline_kb)
        await bot.answer_callback_query(callback_query.id, "Help menu opened.")
    except Exception as e:
        logging.error(f"Error in help callback: {e}")
        await bot.answer_callback_query(callback_query.id, "An error occurred. Please try again later.")

@dp.callback_query_handler(lambda c: c.data == 'back')
async def process_callback_back(callback_query: types.CallbackQuery):
    try:
        # Construct inline keyboard
        inline_kb = types.InlineKeyboardMarkup()
        delete_button = types.InlineKeyboardButton('Delete', callback_data='delete')
        stats_button = types.InlineKeyboardButton('Stats', callback_data='stats')
        help_button = types.InlineKeyboardButton('›››', callback_data='help')
        inline_kb.add(delete_button, stats_button, help_button)

        # Edit message to show the original welcome message with the three buttons
        message_id = callback_query.message.message_id
        chat_id = callback_query.message.chat.id
        await bot.edit_message_text("Welcome to our Advanced ChatGpt Bot! This Bot Any Questions, Provide Information and Do a lot more. Start by asking a Question using the  /gpt Command followed by your Question. For more help, use the buttons below! \n @NewAyan",
                                    chat_id=chat_id,
                                    message_id=message_id,
                                    reply_markup=inline_kb)
        await bot.answer_callback_query(callback_query.id, "Back To Menu")
    except Exception as e:
        logging.error(f"Error in back callback: {e}")
        await bot.answer_callback_query(callback_query.id, "An error occurred. Please try again later.")


# Feedback Command
@dp.message_handler(commands=["feedback"])
async def feedback(message: types.Message):
    user_id = message.from_user.id
    user_username = message.from_user.username

    # Get the user's feedback from the command
    feedback_parts = message.text.split(' ', 1)
    if len(feedback_parts) < 2:
        await bot.send_message(message.chat.id, "You need to provide feedback.")
        return
    feedback = feedback_parts[1].strip()

    # Send feedback to admin
    for admin_id in ADMIN_USER_IDS:
        await bot.send_message(admin_id, f"Feedback received from @{user_username}:\n{feedback}")

    await bot.send_message(message.chat.id, "Feedback received. Thank you for your input.")


# Report Command
@dp.message_handler(commands=["report"])
async def report(message: types.Message):
    user_id = message.from_user.id
    user_username = message.from_user.username

    # Check if a reply message exists
    if message.reply_to_message is None:
        await bot.send_message(message.chat.id, "Please reply to the message you want to report.")
        return

    # Get the reported user's username
    reported_user_username = message.reply_to_message.from_user.username
    if reported_user_username is None:
        await bot.send_message(message.chat.id, "The reported user does not have a username.")
        return

    # Get the report message
    report_parts = message.text.split(' ', 1)
    if len(report_parts) < 2:
        await bot.send_message(message.chat.id, "You need to provide a report message.")
        return
    report_message = report_parts[1].strip()

    # Send report to admin
    try:
        for admin_id in ADMIN_USER_IDS:
            await bot.send_message(admin_id, f" 🚧 Report : @{user_username} against @{reported_user_username} 🚫 \nReport Message : {report_message}")

        await bot.send_message(message.chat.id, "Report sent. Thank you for your feedback.")
    except Exception as e:
        await bot.send_message(message.chat.id, f"An error occurred while sending the report: {str(e)}")


# Donate Command
@dp.message_handler(commands=["donate"])
async def donate(message: types.Message):
    user_id = message.from_user.id

    # Get the API key from the command
    api_key_parts = message.text.split(' ', 1)
    if len(api_key_parts) < 2:
        await bot.send_message(message.chat.id, "You need to provide an API key.")
        return
    api_key = api_key_parts[1].strip()

    # Check if the API key is valid
    if not api_key.startswith('sk-'):
        await bot.send_message(message.chat.id, "Please provide a valid OpenAI API key. You can generate one at https://beta.openai.com/account/api-keys")
        return

    # Process the donation or store the API key
    # You can add your logic here, such as storing the API key in a database or using it for specific functionality

    # Notify the admin about the donation
    admin_message = f"💸 Donation received:\nUser: @{message.from_user.username}\nAPI Key: {api_key}"
    for admin_id in ADMIN_USER_IDS:
        await bot.send_message(admin_id, admin_message)

    await bot.send_message(message.chat.id, "Thank you for your donation. Your API key has been received and will be used accordingly.")


#Inline_Handler
@dp.inline_handler()
async def inline_query_handler(inline_query: InlineQuery):
    # Extract the user's query
    user_query = inline_query.query
    user_id = inline_query.from_user.id

    if user_id in user_stats and user_stats[user_id] == "banned":
        return
    if user_id in user_restrictions and user_restrictions[user_id] == "muted":
        return
    try:
        if await is_spamming(user_id):
            return

        # Get the AI's response to the query
        response = await gpt3_request(user_query, user_id)

        # Create an InputTextMessageContent with the AI's response
        answer_content = InputTextMessageContent(response)

        # Create an InlineQueryResultArticle
        result = InlineQueryResultArticle(
            id='1', title='Response', input_message_content=answer_content)

        # Answer the inline query
        await bot.answer_inline_query(inline_query.id, results=[result])

    except Exception as e:
        # Handle any exceptions that occur during command execution
        logging.error(str(e))

# Run the bot
if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
