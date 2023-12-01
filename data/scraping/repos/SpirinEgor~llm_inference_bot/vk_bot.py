from os import environ

from loguru import logger
from openai import OpenAIError
from vkbottle import Bot, API
from vkbottle.bot import Message
from vkbottle.framework.labeler import BotLabeler

from src.dialogue_tracker import DialogueTracker

_VK_API = API(environ.get("VK_API_TOKEN"))
_VK_BOT_LABELER = BotLabeler()
_DIALOG_TRACKER = DialogueTracker()

_google_spreadsheet_id = environ.get("GOOGLE_SPREADSHEET_ID", None)
if _google_spreadsheet_id is not None:
    from src.google_sheets_wrapper import GoogleSheetsWrapper

    logger.info(f"Using Google Sheets to track usage")
    _GOOGLE_SHEETS_WRAPPER = GoogleSheetsWrapper(_google_spreadsheet_id)
else:
    logger.info(f"No usage tracking")
    _GOOGLE_SHEETS_WRAPPER = None


_HELP_MESSAGE = """Список команд:
- /help -- Помощь
- /role <role> -- Установить кастомную роль
- /reset -- Сбросить роль на стандартную, очистить историю

Максимальное число сообщений в истории: {messages_in_history}, время жизни истории: {max_alive_dialogue} секунд.
Текущая роль: '{role}'
Если бот долго не отвечает, вероятно, OpenAI API перегружено, попробуйте позже.
Если сообщение выводится не до конца, то превышен лимит по токенам, сбросьте историю.
По всем вопросам: @boss
"""

_OPENAI_ERROR_MESSAGE = (
    "Какая-то ошибка на сервере OpenAI, скорее всего перегружен другими запросами 🫠. "
    "Повторите запрос позже или попробуйте сбросить историю с помощью `/reset`"
)
_SYSTEM_ERROR_MESSAGE = (
    "Что-то пошло не так 🫠. Попробуйте сбросить историю с помощью `/reset` или напишите @spirin.egor 🤗!"
)


@_VK_BOT_LABELER.message(command="help")
async def help_message(message: Message):
    user_id = message.from_id
    help_msg = _HELP_MESSAGE.format(role=_DIALOG_TRACKER.get_role(user_id), **_DIALOG_TRACKER.config)
    await message.answer(help_msg)


@_VK_BOT_LABELER.message(command="reset")
async def reset(message: Message):
    user_id = message.from_id
    _DIALOG_TRACKER.reset(user_id)
    await message.answer("Роль сброшена на стандартную, история сброшена")


@_VK_BOT_LABELER.message()
async def handle_message(message: Message):
    user_id = message.from_id
    text = message.text

    if text.startswith("/role"):
        command, argument = text.split(maxsplit=1)
        if not argument:
            await message.answer("Необходимо указать роль: /role <role>")
        _DIALOG_TRACKER.set_role(user_id, argument)
        await message.answer("Роль установлена, история сброшена")
        return

    try:
        answer, total_tokens = await _DIALOG_TRACKER.on_message(message.text, user_id)
        user_info = (await _VK_API.users.get(user_id))[0]
        user_name = f"{user_info.last_name} {user_info.first_name}"

        if _GOOGLE_SHEETS_WRAPPER is not None:
            _GOOGLE_SHEETS_WRAPPER.increase_user_usage(user_id, user_name, total_tokens)
    except OpenAIError as e:
        logger.warning(f"OpenAI API error: {e}")
        answer = _OPENAI_ERROR_MESSAGE
    except Exception as e:
        logger.error(e)
        answer = _SYSTEM_ERROR_MESSAGE
    await message.answer(answer)


def main():
    logger.disable("vkbottle")
    logger.info(f"Starting VK bot")
    bot = Bot(api=_VK_API, labeler=_VK_BOT_LABELER)
    bot.run_forever()


if __name__ == "__main__":
    main()
