from aiogram import Bot, Dispatcher, executor, types
from config import ConfigManager
import logging
from bot.openai_api import OpenAIManager
from aiogram.utils.exceptions import TelegramAPIError, CantParseEntities

class MyChatBot:
    def __init__(self, telegram_bot_token, openai_api_key):
        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Create bot and dispatcher instances

        self.bot = Bot(token=telegram_bot_token)
        self.dp = Dispatcher(self.bot)

        # Create OpenAI manager instance
        self.openai_manager = OpenAIManager(openai_api_key)

        # Define message history as an instance variable
        self.message_history = [{"role": "system", "content": "Вы - доброжелательный и отзывчивый помощник преподавателя. Вы глубоко объясняете понятия, используя простые термины, и приводите примеры, чтобы помочь людям учиться. В конце каждого объяснения вы задаете вопрос, чтобы проверить понимание."}]

        # Register message handlers
        self.dp.register_message_handler(self.send_welcome, commands=['start'])
        self.dp.register_message_handler(self.send_help, commands=['help'])
        self.dp.register_message_handler(self.new_conversation, commands=['new'])
        self.dp.register_message_handler(self.echo)


    async def setup_bot_commands(self, dispatcher: Dispatcher):
        commands = [
            types.BotCommand(command="/start", description="Запустить бота"),
            types.BotCommand(command="/help", description="Получить справку"),
            types.BotCommand(command="/new", description="Начните новый разговор"),
        ]
        await dispatcher.bot.set_my_commands(commands)

    async def send_welcome(self, message: types.Message):
        await message.reply("Привет! Я бот, с которым можно общаться. Я использую GPT-3 от OpenAI для генерации ответов. Напишите /help, чтобы узнать больше.")
    
    async def new_conversation(self, message: types.Message):
        self.message_history = [{"role": "system", "content": "Вы - доброжелательный и отзывчивый помощник преподавателя. Вы глубоко объясняете понятия, используя простые термины, и приводите примеры, чтобы помочь людям учиться. В конце каждого объяснения вы задаете вопрос, чтобы проверить понимание."}]
        await message.reply("Давайте начнем новый разговор.")

    async def send_help(self, message: types.Message):
        """Send a message when the command /help is issued."""
        text = (
            "Привет! Я бот, с которым можно общаться. Я использую GPT-3 от OpenAI для генерации ответов. "
            "Вот некоторые вещи, которые вы можете попросить меня сделать:\n\n"
            "📅 Запланировать свой день\n"
            "🍝 Найти рецепт\n"
            "💸 Разделить бюджет\n"
            "📝 Написать стихи или историю\n"
            "🎶 Сгенерировать тексты песен\n"
            "🎨 Создать задания для художников\n"
            "📚 Найти рекомендации книг\n"
            "🗺️ Сгенерировать план путешествия\n"
            "🎬 Придумать идею для фильма\n"
            "🎲 Сгенерировать случайные идеи\n\n"
            "Используйте /new, чтобы начать новую беседу.\n"
            "Используйте /help, чтобы увидеть это сообщение снова. "
        )
        await message.reply(text)

    async def echo(self, message: types.Message):
        try:
            user_input = message.text
            if len(user_input) > 2048:
                await message.answer("Сообщение слишком длинное.")
                raise ValueError("Message is too long.")
            self.message_history.append({"role": "user", "content": user_input})

            num_tokens = sum(len(msg["content"].split()) for msg in self.message_history)
            if num_tokens > 3500:
                self.message_history = self.clean_messages()
                await message.answer("Сообщение слишком длинное.")
                raise ValueError("Message is too long.")
            await self.bot.send_chat_action(message.chat.id, 'typing')

            # Get OpenAI response
            openai_response = await self.openai_manager.get_openai_response(self.message_history)
            self.message_history.append({"role": "assistant", "content": openai_response})

            # Send message with different parse modes until one works
            parse_modes = ['MarkdownV2', 'Markdown', 'HTML']
            for parse_mode in parse_modes:
                try:
                    await message.answer(openai_response, parse_mode=parse_mode)
                    break
                except CantParseEntities:
                    if parse_mode == parse_modes[-1]:
                        raise

        except (TelegramAPIError, ValueError) as e:
            logging.error(f"Error processing message: {str(e)}")
            error_message = "Извините, что-то пошло не так. Попробуйте снова. Если ошибка повторится, пожалуйста, сообщите об этом."
            self.message_history = self.clean_messages()
            await message.answer(error_message)

        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            self.message_history = self.clean_messages()

    def clean_messages(self):
        first_message = "Вы - доброжелательный и отзывчивый помощник преподавателя. Вы глубоко объясняете понятия, используя простые термины, и приводите примеры, чтобы помочь людям учиться. В конце каждого объяснения вы задаете вопрос, чтобы проверить понимание."
        message_history = [{"role": "system", "content": first_message}]
        return message_history

    def start(self):
        executor.start_polling(self.dp, skip_updates=True, on_startup=self.setup_bot_commands)
