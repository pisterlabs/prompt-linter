"""
В этом файле самая жесть. Тут сама логика дяалога с ГПТ
"""
from pprint import pprint
from typing import Any
from httpx import Response

from aiogram.fsm.context import FSMContext
from aiogram.methods import SendMessage
from aiogram import Bot, types
from sqlalchemy.orm import sessionmaker

from bot import settings as sett
import bot.db as db
from bot import openai_async
from ._tools import generate_payload, get_api_key, check_tokens_buffer, \
    add_message, set_accounting, message_handle_fn, gen_dialogue_cache, \
    rename_dialogue
from bot.handlers._accounting import check_tokens_limit
from bot.handlers.states import DialogueStates
import bot.handlers.keyboards.user_kb as ukb


async def new_GPT_3(
        message: types.Message,
        state: FSMContext,
        session_maker: sessionmaker
        ):
    """
    Создаёт новый диалог с GPT_3 и открывает его, заполняя контекстом
    """
    await state.clear() # на всякий случай чистит состояние
    msg = await message.answer(
        'Секунду...',
        )
    user_id = message.from_user.id
    await state.set_state(DialogueStates.dialogue) # выставляет состояние диалога

    last_dialogue = await db.create_dialogue( # создаёт диалог в БД
        user_id,
        session_maker,
        'gpt-3.5-turbo'
        )
    await gen_dialogue_cache(
        user_id,
        state,
        last_dialogue,
        session_maker)
    await check_tokens_limit(
        state,
        session_maker
        )
    await msg.edit_text(
        f'Диалог {last_dialogue.name} создан.',
        )
    

async def new_GPT_4(
        message: types.Message,
        state: FSMContext,
        session_maker: sessionmaker
        ):
    """
    Создаёт новый диалог с GPT_4 и открывает его, заполняя контекстом
    """
    await state.clear()
    msg = await message.answer(
        'Секунду...',
        )
    user_id = message.from_user.id
    last_dialogue = await db.create_dialogue(user_id, session_maker, 'gpt-4')
    await db.get_or_create_account(user_id, 'gpt-4', session_maker) # 
    await state.set_state(DialogueStates.dialogue)
    await generate_payload(state, last_dialogue, session_maker)
    await set_accounting(state, user_id, session_maker)
    await check_tokens_limit(
        state,
        session_maker
        )
    await state.update_data(api_key=sett.GPT4_API_KEY)
    await msg.edit_text(f'Диалог {last_dialogue.name} создан.')


async def new_dial_with_prompt(
        query: types.CallbackQuery,
        callback_data: ukb.SelectPromptCD,
        state: FSMContext,
        session_maker: sessionmaker
        ):
    """
    Создаёт новый диалог на основе промпта из БД
    """
    prompt = await db.prompt_get_by_id(int(callback_data.id), session_maker)
    user_id = query.from_user.id
    last_dialogue = await db.create_dialogue(
        user_id,
        session_maker,
        'gpt-3.5-turbo',
        parse_mode=prompt.parse_mode,
        name=prompt.name
        )
    await db.get_or_create_account(user_id, 'gpt-3.5-turbo', session_maker) #
    await state.set_state(DialogueStates.dialogue)
    await generate_payload(state, last_dialogue, session_maker)
    await set_accounting(state, user_id, session_maker)
    await SendMessage(
        chat_id=user_id,
        text=prompt.welcome_message,
        parse_mode=prompt.parse_mode
    )


async def open_dialogue(
        query: types.CallbackQuery,
        callback_data: ukb.UserDialoguesCallback,
        state: FSMContext,
        session_maker: sessionmaker
    ):
    """
    Открывает диалог по кнобке из списка диалогов
    """
    dial = await db.get_dial_by_id(callback_data.dial_id, session_maker)
    await state.set_state(DialogueStates.dialogue)
    await generate_payload(state, dial, session_maker)
    await get_api_key(dial, state)
    await set_accounting(state, query.from_user.id, session_maker) 
    return await SendMessage(
        text=f'Диалог {dial.name} открыт',
        chat_id=query.from_user.id,
        )


async def dialogue(
        message: types.Message,
        state: FSMContext,
        session_maker: sessionmaker
        ):
    """
    Тут вся логика диалога
    """
    data = await add_message(message.text, "user", state)
    dialogue_id = data['dialogue_id']
    payload = data.get('payload')
    model = payload.get("model")

    await check_tokens_limit(state, session_maker)
    await db.create_message( # создаёт сообщение из текста присланного юзером
        dialogue_id,
        role='user',
        text=message.text,
        session_maker=session_maker
        )
    await check_tokens_buffer(state, session_maker)

    msg = await SendMessage(
        text=f'Запрос к {model} отправлен',
        chat_id=message.from_user.id,
    )
    chat_response = await message_handle_fn(state, msg)
    data = await add_message(chat_response, "assistant", state)
    dial_name: str = data.get('dialogue_name')

    await check_tokens_limit(state, session_maker)
    await db.create_message( # сохраняет в БД ответ ГПТ
        dialogue_id,
        role='assistant',
        text=chat_response,
        session_maker=session_maker
        )
    await check_tokens_buffer(state, session_maker)

    if dial_name.startswith(model):
        await rename_dialogue(dialogue_id, state, session_maker)


async def transcribe_to_gpt(
        query: types.CallbackQuery,
        callback_data: ukb.TranscribeCD,
        state: FSMContext,
        session_maker: sessionmaker,
        **data: dict[str, Any],
    ):
    """
    От кнопки "Отправить ГПТ" получает айди сообщения с данными
    и отправляет его ГПТ
    """
    mess_id = callback_data.message_id
    chat_id = query.from_user.id
    bot: Bot = data['bot']

    mess = await bot.forward_message(chat_id, chat_id, mess_id)
    cache_data = await add_message(mess.text, "user", state)
    payload = cache_data.get('payload')
    await db.create_message(
        cache_data['dialogue_id'],
        role='user',
        text=mess.text,
        session_maker=session_maker
        )
    await mess.delete()
    await check_tokens_buffer(state, session_maker)
    msg = await SendMessage(
        text=f'Запрос к {payload.get("model")} отправлен',
        chat_id=query.from_user.id,
    )
    completion: Response = await openai_async.chat_complete(
        api_key=cache_data.get('api_key'),
        timeout=60,
        payload=payload,
        )
    try:
        chat_response = completion.json()["choices"][0]["message"]['content']
    except KeyError:
        return await msg.edit_text('Что-то пошло не так')
    
    cache_data = await add_message(chat_response, "assistant", state)
    await msg.edit_text(chat_response)
    await db.create_message(
        cache_data['dialogue_id'],
        role='assistant',
        text=chat_response,
        session_maker=session_maker
        )
    await check_tokens_buffer(state, session_maker)
    payload = cache_data.get('payload')
