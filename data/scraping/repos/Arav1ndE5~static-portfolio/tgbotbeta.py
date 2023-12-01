from collections import deque
import openai
import telebot
import re
import time


openai.api_key = "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"

# Set up Telegram bot credentials
telegram_bot_token = "765669552:AAEEl54GpxoFu0SvHVxX6aK31lYUpnhAtts"
bot = telebot.TeleBot(telegram_bot_token)

# Initialize conversation history
chat_history = []
max_chat_history = 4
chat_history = deque(maxlen=max_chat_history)

# Greeting and other patterns (unchanged)
greeting_pattern = re.compile(r"^(hi|hai|hei|oii|oi|hello|hey|greetings|yo)\b", re.IGNORECASE)
owner_pattern = re.compile(r"(creator of bot|owner of bot|bot owner|owns this bot|created the bot|created you|made you|your father|your owner|your creator|who runs you)\b", re.IGNORECASE)
thanks_pattern = re.compile(r"^(thanks|thank you|appreciate it|much obliged)\b", re.IGNORECASE)
bot_pattern = re.compile(r"^(your name|who are you|what are you)\b", re.IGNORECASE)
main_pattern = re.compile(r"(Aravindee|Seban|AR7|Arav1nd|Aravind es|master aravind|Aravind E S|Aravindeee|aravindes)\b", re.IGNORECASE)
new_pattern = re.compile(r"(Mariya|mariya joy|maria|mariyamma|Aravind love|love mariya|aravind es love|mariya love)\b", re.IGNORECASE)
end_pattern = re.compile(r"^(bye|buei|babye|bui|bei|bie)\b", re.IGNORECASE)

# Define the /start command handler (unchanged)
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello, how is your day for more click /help")

@bot.message_handler(commands=['help'])
def help(message):
    bot.reply_to(message, "Select cheyy: \n /start \n /game \n /help \n /status")

@bot.message_handler(commands=['game'])
def info(message):
    bot.reply_to(message, "https://aravind-es.w3spaces.com/")

@bot.message_handler(commands=['status'])
def status(message):
    bot.reply_to(message, "Created by Arav1nd  \n https://aravindes.onrender.com/")

# Define the function to handle user messages
@bot.message_handler(func=lambda message: True)
def chat(message):
    # Check if the message author is the bot itself
    if message.from_user.id == bot.get_me().id:
        return
    if message.from_user.id == "":
        bot.reply_to(message,"hmmm")
        return

    # Check for greetings and thanks
    if greeting_pattern.search(message.text):
        bot.reply_to(message, "Heyy 🙂")
        return
    elif thanks_pattern.search(message.text):
        bot.reply_to(message, "thanks!")
        return
    elif owner_pattern.search(message.text):
        bot.reply_to(message, "Master Aravind?.")
        return
    elif new_pattern.search(message.text):
        bot.reply_to(message, "Yes you, so much")
        return
    elif main_pattern.search(message.text):
        bot.reply_to(message, "Aravind created me. That is what i am told to answer.")
        return
    elif bot_pattern.search(message.text):
        bot.reply_to(message, "I am AR7. half Arifical Intelligence half No Intelligence.🫠")
        return
    elif end_pattern.search(message.text):
        bot.reply_to(message, "Ok bye. Will miss you.")
        return

    # Get user input
    user_input = message.text

    # Construct prompt with conversation history
    prompt = "You are AR7. You are funny and flirty. Try to keep answers short and make jokes. Answer all queries. \n".join(chat_history) +"\n User:" + user_input + "\nAR7:"

    while True:
        try:
            # Generate AI response
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=1,
                max_tokens=60,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                stop=None
            )
            break  # Exit the retry loop if the API call is successful
        except openai.error.RateLimitError:
            # Wait for 20 seconds and then retry the API call
            time.sleep(20)

    # Get the generated response text
    response_text = response.choices[0].text.strip()

    chat_history.append("User:" + user_input)
    chat_history.append("AR7:" + response_text)
    
    print(chat_history)

    # Check if the response starts with self declaration
    if response_text.startswith("AR7:"):
        # Remove the question mark
        response_text = response_text[4:].strip()

    if response_text:
        bot.reply_to(message, response_text)
    else:
        bot.reply_to(message, "😅🙂")

# Start the bot
print("oooduuvaaaaaaaaa")
bot.polling()
