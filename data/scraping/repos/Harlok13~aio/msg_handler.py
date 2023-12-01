import logging

from aiogram import Router, F
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot_ai.states.states import FSMOpenaiModel
from bot_ai.utils.bot_models import OPENAI_MODELS
from bot_ai.utils.current_state_info import get_current_state_info
from bot_ai.utils.user_requsts import UserRequest

router = Router()
logger = logging.getLogger(__name__)


# ref
@router.message(StateFilter(FSMOpenaiModel.set_standard))
async def default_model_answer(message: Message, request: UserRequest, state: FSMContext) -> None:
    print(current_state := await state.get_state(), 'default state')
    await message.reply('sec')
    if check_user_tokens := await request.check_user_tokens(message.from_user.id):
        response, msg = OPENAI_MODELS.get(get_current_state_info(current_state))(message)
        await request.increase_question_count(user_id=message.from_user.id)

        logger.info(message.text)
        logger.info(message.from_user.first_name)
        logger.info(response.choices[0].text)

        await message.answer(msg)
    else:
        await message.answer('У вас недостаточно токенов')


# ref
@router.message(StateFilter(FSMOpenaiModel.set_companion))
async def companion_model_answer(message: Message, request: UserRequest, state: FSMContext) -> None:
    print(current_state := await state.get_state(), 'state companion')

    await message.reply('sec')

    if check_user_tokens := await request.check_user_tokens(message.from_user.id):
        response, msg = OPENAI_MODELS.get(get_current_state_info(current_state))(message)
        await request.increase_question_count(user_id=message.from_user.id)

        logger.info(message.text)
        logger.info(message.from_user.first_name)

        total_openai_spent_tokens: int = response.usage.total_tokens
        await request.decrease_user_tokens(
            user_id=message.from_user.id,
            openai_spent_tokens=total_openai_spent_tokens
        )
        await message.answer(msg)
    else:
        await message.answer('У вас недостаточно токенов')
