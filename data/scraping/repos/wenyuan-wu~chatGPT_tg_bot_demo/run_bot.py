import os
import telebot
from openai_res import get_response_openai, get_response_openai_test
from dotenv import load_dotenv

max_turns = 20
conversations = {}  # Dictionary to store the conversation history for each user


def run_tg_bot(bot_token):
    """
    Function to initialize the bot.
    :param bot_token: environment variable BOT_TOKEN
    :return: None
    """
    bot = telebot.TeleBot(bot_token)

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        text = "Hello, this is a tiny demo for a ChatGPT-like conversational agent.\nSimply type any message to get " \
               "a AI generated reply.\ne.g. 'What is quantum computing?'"
        bot.send_message(message.chat.id, text, parse_mode="Markdown")

    @bot.message_handler(func=lambda msg: True)
    def respond_openai(message):
        chat_id = message.chat.id
        global conversations, max_turns

        # Initialize conversation history for new user
        if chat_id not in conversations:
            conversations[chat_id] = []
            conversations[chat_id].append({"role": "system", "content": "You are a helpful assistant."})

        # Start new conversation if user types /new
        if message.text == "/new":
            conversations[chat_id] = []
            conversations[chat_id].append({"role": "system", "content": "You are a helpful assistant."})
            bot.send_message(chat_id, "Start a new conversation.")
        else:
            # Add user message to conversation history
            conversations[chat_id].append({"role": "user", "content": message.text})

            # Get AI generated response
            prompt = conversations[chat_id]
            reply = get_response_openai(prompt)

            # Add AI generated response to conversation history
            conversations[chat_id].append({"role": "assistant", "content": reply})

            # Send AI generated response to user
            bot.reply_to(message, reply)

            # Check if maximum turns has been reached
            if len(conversations[chat_id]) // 2 >= max_turns:
                # Send message to user to indicate maximum turns has been reached
                bot.send_message(chat_id,
                                 "This conversation has ended as the maximum number of turns has been reached.")

                # Reset conversation history for this user
                conversations[chat_id] = []
            elif len(conversations[chat_id]) // 2 >= max_turns - 5:
                remaining_turns = max_turns - len(conversations[chat_id]) // 2
                bot.send_message(chat_id, f"{remaining_turns} turns left in this conversation. "
                                          f"To start a new conversation, type /new.")

    bot.infinity_polling()


def main():
    load_dotenv()
    bot_token = os.environ.get('BOT_TOKEN')
    run_tg_bot(bot_token)


if __name__ == '__main__':
    main()
