import openai
import telebot
import random

ERROR_MESSAGES = [
    "I'm sorry, Akigpt couldn't come up with a response. Please try again.",
    "Akigpt is helping Aki with making sushi right now. Please try again later",
    "Akigpt went to bathroom. Please try again later.",
]

# Define OpenAI and telegram API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
bot = telebot.TeleBot((os.environ.get("TELEGRAM_BOT_TOKEN"))


# Define bot personality
BOT_PERSONALITY = {
    "temperature": 1,
    "max_tokens": 60,
    "top_p": 1,
    "presence_penalty": 0.9,
    "frequency_penalty": 0.9,
    "stop": "\n",
}


 #Define OpenAI prompt for additional context
OPENAI_PROMPT = "My name is Akigpt, and I will give you useful information and entertain you. How can I help you?"


@bot.message_handler(commands=['start'])
def main(message):
     msg = bot.send_message(message.chat.id, "Hi, I'm Akigpt. Ask me anything.")
     bot.register_next_step_handler(msg, chatgpt)

@bot.message_handler(content_types=['text'])
def chatgpt(message):
    prompt = f"User: {message.text}\nBot:"
    completion = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        **BOT_PERSONALITY
   )

    response = completion.choices[0].text.strip()

    # Debugging statement
    print(f"Response: {response}")

    # random error messsage
    if not response:
        error_message = random.choice(ERROR_MESSAGES)
        bot.send_message(message.chat.id, error_message)
    else:
        # Send response
        bot.send_message(message.chat.id, response)

bot.polling(none_stop=True)
