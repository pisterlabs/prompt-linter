from aiogram import types, Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from infrastructure.openai_api.api import OpenAIAPIClient
from tgbot.filters.admin import AdminFilter
from tgbot.services.assistant import ChatAssistant

calendar_router = Router()
calendar_router.message.filter(AdminFilter())

MAX_STEPS = 5


@calendar_router.message(Command("reset"))
async def reset(message: types.Message, state: FSMContext):
    await state.clear()
    await message.reply("Conversation was reset")


@calendar_router.message(F.text)
async def any_message(
    message: types.Message,
    state: FSMContext,
    openai: OpenAIAPIClient,
    calendar_service,
    function_definitions,
):
    assistant = ChatAssistant(
        message, state, openai, calendar_service, function_definitions
    )

    await assistant.process_message()
