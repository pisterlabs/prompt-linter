import openai
from aiogram import Router
from aiogram.enums import ContentType
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.filters import StateFilter

from keyboards.cancel_keyboard import get_cancel_keyboard
from models.user import User
from states import Global

from main import server
from utils.file_data import FileData
from utils.filter import CallbackFilter

router = Router()


@router.callback_query(CallbackFilter(data='non-context-action.edit-image'), StateFilter(None))
async def on_edit_started(cb: CallbackQuery, state: FSMContext):
    user = User.get_by_callback(cb)

    await cb.answer()
    await cb.message.answer(user.get_string("non-context-action.edit-image.prompt_ask"), reply_markup=get_cancel_keyboard(user))
    await state.set_state(Global.image_editing)


@router.message(Global.image_editing)
async def on_edit(msg: Message, state: FSMContext):
    user = User.get_by_message(msg)

    if msg.content_type != ContentType.TEXT and msg.content_type != ContentType.PHOTO:
        await msg.answer(user.get_string("non-context-action.edit-image.invalid_content_type"), reply_markup=get_cancel_keyboard(user))
        return
    if msg.content_type == ContentType.TEXT:
        data = await state.get_data()
        data["prompt"] = msg.text
        await state.set_data(data)
        await msg.answer(user.get_string("non-context-action.edit-image.prompt-set"))
    if msg.content_type == ContentType.PHOTO:
        data = await state.get_data()
        data["image"] = FileData(msg.photo[-1]).get_data()
        await state.set_data(data)
        await msg.answer(user.get_string("non-context-action.edit-image.image-set"))

    data = await state.get_data()
    if "image" in data.keys() and "prompt" in data.keys():
        new_msg = await msg.answer(user.get_string("non-context-action.edit-image.image-downloading"))

        file = await user.download_file_by_id(data["image"], "png")

        await new_msg.edit_text(user.get_string("non-context-action.edit-image.in-process"))
        try:
            await state.set_state(Global.busy)
            response = await openai.Image.acreate_edit(
                image=open(file, 'rb'),
                prompt=data["prompt"],
                n=1,
                size="1024x1024"
            )
            image_url = response['data'][0]['url']
            await new_msg.delete()
            await msg.answer_photo(photo=image_url)
        finally:
            await server.delete_file(file)
            await state.clear()
