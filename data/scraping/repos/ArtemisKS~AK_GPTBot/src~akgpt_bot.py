import openai
import logging
import time
from telegram import Update, ReplyKeyboardRemove, BotCommandScopeDefault, BotCommandScopeAllChatAdministrators, Bot, BotCommand, error
from telegram.ext import (
    Updater,
    CallbackQueryHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)

from financial_validator import FinancialValidator
from message_limit_handler import MessageLimitHandler
from input_handler import InputHandler
from localization import loc, translator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GPTBot:
    def __init__(self, telegram_api_key, gpt_api_key):
        self.TELEGRAM_API_KEY = telegram_api_key
        self.GPT_API_KEY = gpt_api_key

        self.bot = Bot(token=telegram_api_key)
        self.updater = Updater(self.TELEGRAM_API_KEY)
        self.input_handler = InputHandler(MessageLimitHandler(), FinancialValidator(), self.updater)
        
        # Initialize OpenAI API
        openai.api_key = self.GPT_API_KEY
        
        self.non_admin_commands = ["start", "gpt", "help"]
        self.admin_commands = self.non_admin_commands + ["adminmenu"]
        
        self.setup_commands_methods()
        
    def handle_retry_after(self, update: Update, context: CallbackContext):
        try:
            raise context.error
        except error.RetryAfter as e:
            logging.warning(f"Caught RetryAfter error: {e}, waiting for {e.retry_after} seconds before retrying.")
            # Optional: Send a warning message to the user
            reply = loc('flood_control', seconds=e.retry_after)
            context.bot.send_message(chat_id=update.effective_chat.id, text=reply)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        
    def handle_command(self, update: Update, context: CallbackContext):
        if update.message is None or update.message.text is None:
            # Send a message to the user that the bot only processes text messages
            context.bot.send_message(chat_id=update.effective_chat.id, text=loc('only_text_messages'))
            return
        command_with_args = update.message.text.split()
        full_command = command_with_args[0][1:]  # Extract the command without the leading '/'
        command = full_command.split('@')[0]  # Remove the bot's username if it's present
        if command in self.commands_methods:
            language_code = self.get_user_language(update)
            
            new_language = not translator.is_current_lang(language_code)
            if new_language:
                chat_id = update.effective_chat.id
                self.input_handler.start_typing(context, chat_id)
                
            translator.change_language(language_code)
            
            if new_language:
                self.update_commands()
            
            self.input_handler.stop_typing()
            
            if command == 'gpt':
                # Extract the arguments
                args = command_with_args[1:]
                self.commands_methods[command](update, context, args)
            else:
                self.commands_methods[command](update, context)
        else:
            self.unknown_command(update)

    def start(self, update: Update, context: CallbackContext):
        update.message.reply_text(loc('greeting'), reply_markup=ReplyKeyboardRemove())

    def help_command(self, update: Update, context: CallbackContext):
        # Check if the user is an admin
        is_admin = self.input_handler.is_user_admin(update, context)

        commands = self.admin_commands if is_admin else self.non_admin_commands

        help_text = '\n'.join([f'/{cmd} - {loc(cmd)}' for cmd in commands])

        help_text = f"{loc('available_commands')}:\n{help_text}"
        update.message.reply_text(help_text)

    def unknown_command(self, update: Update):
        update.message.reply_text(loc('unknown_command'))
        
    def set_bot_commands_with_retry(self, commands, scope, retries=3, delay=5):
        for attempt in range(retries):
            try:
                self.set_bot_commands(commands, scope)
                return
            except Exception:
                if attempt < retries - 1:  # No need to sleep for the last attempt
                    time.sleep(delay)
                else:
                    logging.error(f"Failed to set bot commands after {retries} attempts due to timeout.")
                    return
        
    def set_bot_commands(self, commands, scope):
        self.bot.set_my_commands(commands=commands, scope=scope)
        
    def setup_commands_methods(self):
        self.commands_methods = {
            'start': self.start,
            'gpt': self.input_handler.gpt,
            'help': self.help_command,
            'adminmenu': self.input_handler.show_admin_menu,
        }
        
    def update_commands(self):
        
         # Convert commands dictionaries to lists of BotCommand objects
        non_admin_bot_commands = [BotCommand(cmd, loc(cmd)) for cmd in self.non_admin_commands]
        admin_bot_commands = [BotCommand(cmd, loc(cmd)) for cmd in self.admin_commands]

        # Set commands for non-admin users
        default_scope = BotCommandScopeDefault(type='default')
        self.set_bot_commands_with_retry(non_admin_bot_commands, default_scope)

        # Set commands for admin users
        admin_scope = BotCommandScopeAllChatAdministrators(type='all_chat_administrators')
        self.set_bot_commands_with_retry(admin_bot_commands, admin_scope)
        
    def get_user_language(self, update: Update):
        lang_code = update.effective_user.language_code
        return lang_code

    def run(self):
        dp = self.updater.dispatcher

        # Register the common command handler
        dp.add_handler(MessageHandler(Filters.command, self.handle_command))
        dp.add_handler(CallbackQueryHandler(self.input_handler.handle_admin_callback))
        dp.add_handler(MessageHandler(Filters.text & (~Filters.command), self.input_handler.handle_text))
        dp.add_error_handler(self.handle_retry_after)

        self.updater.start_polling()
        self.updater.idle()


# Set your API keys as environment variables
TELEGRAM_API_KEY = 'tg_api_key' #os.getenv('TELEGRAM_API_KEY')
GPT_API_KEY = 'gpt_api_key' #os.getenv('GPT_API_KEY')

if __name__ == '__main__':
    gpt_bot = GPTBot(TELEGRAM_API_KEY, GPT_API_KEY)
    gpt_bot.run()
