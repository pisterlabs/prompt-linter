import openai
from googletrans import Translator
from aiogram import Bot, Dispatcher, executor, types

"""
Example with a counter instead of an instant ban
"""

class AutoModeration:
    translator = Translator()

    def __init__(self, openai_token: str, ban_words: list, ban: bool, language: str) -> None:
        self.openai_token = openai_token
        self.ban_words = ban_words
        self.ban = ban
        self.language = language

    def gen_context_msg_gpt(self, msg: str, ban_words: str) -> str:
        if self.ban_words:
            return f"""Hi, read this message\n{msg} and if it contains at least one word of their list - {ban_words}\nAlso, do you think this message is spam?, say yes or no"""
        else:
            return """Determine whether this message is spam or not, if yes, write yes in the answer"""

    def send_question_chatgpt(self, msg: str) -> bool:
        if self.language == "ru":
            content_to_chatgpt = self.translator.translate(self.gen_context_msg_gpt(msg, self.ban_words), src="ru", dest="en").text
        else:
            content_to_chatgpt = self.gen_context_msg_gpt(msg, self.ban_words)
        openai.api_key = self.openai_token
        messages = [{"role": "user", "content": content_to_chatgpt}]
        chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        response_from_chatgpt = chatgpt_response["choices"][0]["message"]["content"]
        return "Да" in response_from_chatgpt.lower() or "Yes" in response_from_chatgpt.lower() or "contains" in response_from_chatgpt.lower()


class TelegramBot:
    ban_counter = {}

    def __init__(self, telegram_token:str, counter_ban:int) -> None:
        self.telegram_token = telegram_token
        self.counter_for_ban = counter_ban

    def start(self):
        bot = Bot(token=self.telegram_token)
        dp = Dispatcher(bot)

        @dp.message_handler(chat_type=types.ChatType.SUPERGROUP)
        async def spam_handler_supergroup(msg: types.Message):
            user_id = msg["from"]["id"]
            chat_id = msg["chat"]["id"]
            msg_id = msg["message_id"]
            moderation_class = AutoModeration("open ai token", ["bruh"], True, "ru")
            is_spam = moderation_class.send_question_chatgpt(msg.text)
            if is_spam and moderation_class.ban:
                await bot.delete_message(chat_id, msg_id)
                if user_id not in self.ban_counter:
                    self.ban_counter[user_id] = 1
                else:
                    self.ban_counter[user_id] = self.ban_counter[user_id] + 1
                if self.ban_counter[user_id] == self.counter_for_ban:
                    await bot.ban_chat_member(chat_id, user_id)
            elif is_spam:
                await bot.delete_message(chat_id, msg_id)

        executor.start_polling(dp)

if __name__ == "__main__":
    TelegramBot("telegram bot token", 3).start()