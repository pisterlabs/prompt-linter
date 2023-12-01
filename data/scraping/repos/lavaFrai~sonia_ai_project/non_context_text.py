import openai
import tiktoken as tiktoken
from aiogram import Router, F
from aiogram.enums import ContentType, ChatAction
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, FSInputFile

from keyboards.non_context_action import get_non_context_text_keyboard
from main import server
from models.user import User
from states import Global
from utils.openai_utils import chatgpt_generate_one_message, openai_text_to_speech

router = Router()


@router.message(lambda x: x.content_type == ContentType.TEXT, StateFilter(None), lambda x: not x.text.startswith('/'))
async def on_non_context_text(msg: Message):
    user = User.get_by_message(msg)

    await msg.reply(user.get_string("non-context-action.non_context_text.action-select"),
                    reply_markup=await get_non_context_text_keyboard(msg.text, user))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string.
        For text-davinci-003 encoding is p50k_base, for newest cl100k_base"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


@router.callback_query(F.data.startswith("non_context_text.continue"), StateFilter(None))
async def on_non_context_text_continue(cb: CallbackQuery, state: FSMContext):
    user = User.get_by_callback(cb)

    source_message = cb.message.reply_to_message
    if source_message is None:
        await cb.answer()
        await cb.message.answer(user.get_string("non-context-action.non_context_voice.message-error"))
        return
    else:
        await state.set_state(Global.busy)
        try:
            message_length_in_tokens = num_tokens_from_string(source_message.text, "p50k_base")
            # await source_message.reply(text=str(message_length_in_tokens))
            completion = openai.Completion.acreate(
                model="text-davinci-003",
                prompt=source_message.text,
                max_tokens=4097 - message_length_in_tokens,
                temperature=0.3
            )
            completion = await server.await_with_typing_status(completion, cb.message.chat.id)

            text = source_message.text + completion["choices"][0]["text"]
            await source_message.reply(text=text)
            # await new_msg.reply(text=server.get_string("non-context-action.non_context_text.additional-action"),
            #                     reply_markup=await get_non_context_text_keyboard(text))
        finally:
            await state.clear()
            await cb.answer()


@router.callback_query(F.data.startswith("non_context_text.reduce"), StateFilter(None))
async def on_non_context_text_reduce(cb: CallbackQuery, state: FSMContext):
    user = User.get_by_callback(cb)

    source_message = cb.message.reply_to_message
    if source_message is None:
        await cb.answer()
        await cb.message.answer(user.get_string("non-context-action.non_context_voice.message-error"))
        return
    else:
        await state.set_state(Global.busy)
        try:
            # message_length_in_tokens = num_tokens_from_string(source_message.text, "p50k_base")
            # new_msg = await source_message.reply(text=user.get_string("generation-in-progress"))

            text = chatgpt_generate_one_message(
                "You should shorten the texts that are sent to you, leaving only the most important in the text. "
                "The result must be in the same language as the original.",
                source_message.text
            )
            text = await server.await_with_typing_status(text, cb.message.chat.id)

            # await new_msg.delete()
            await source_message.reply(text=text, reply_markup=await get_non_context_text_keyboard(text, user))
        finally:
            await state.clear()
            await cb.answer()


@router.callback_query(F.data.startswith("non_context_text.check_grammar"), StateFilter(None))
async def on_non_context_text_reduce(cb: CallbackQuery, state: FSMContext, user: User):
    source_message = cb.message.reply_to_message
    if source_message is None:
        await cb.answer()
        await cb.message.answer(user.get_string("non-context-action.non_context_voice.message-error"))
        return
    else:
        await state.set_state(Global.busy)
        try:
            # message_length_in_tokens = num_tokens_from_string(source_message.text, "p50k_base")
            # new_msg = await source_message.reply(text=server.get_string("generation-in-progress"))

            text = chatgpt_generate_one_message(
                "Correct the spelling, syntax and grammar of this text in source language of message. "
                "Write anything and it will correct your Spelling and Grammar.",
                source_message.text
            )
            text = await server.await_with_typing_status(text, cb.message.chat.id)

            # await new_msg.delete()
            await source_message.reply(text=text, reply_markup=await get_non_context_text_keyboard(text, user))
        finally:
            await state.clear()
            await cb.answer()


@router.callback_query(F.data.startswith("non_context_text.vocalize"), StateFilter(None))
async def on_non_context_text_vocalize(cb: CallbackQuery, state: FSMContext, user: User):
    source_message = cb.message.reply_to_message
    if source_message is None:
        await cb.answer()
        await cb.message.answer(user.get_string("non-context-action.non_context_voice.message-error"))
        return
    await state.set_state(Global.busy)

    text = cb.message.reply_to_message.text
    voice_file = await server.create_file('mp3')

    make_speech = openai_text_to_speech(text, voice_file)
    await server.await_with_typing_status(make_speech, cb.message.chat.id, ChatAction.RECORD_VOICE)
    upload_speech = await cb.message.reply_to_message.reply_voice(voice=FSInputFile(voice_file))
    # await server.await_with_typing_status(upload_speech, cb.message.chat.id, ChatAction.UPLOAD_VOICE)

    await cb.answer()
    await state.clear()
