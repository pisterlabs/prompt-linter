from typing import Tuple, Dict, Optional

import tempfile
import asyncio
import pydub
import logging
from datetime import datetime
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
from telegram.constants import ParseMode, ChatType

from bot import openai_utils

from bot import config
from bot.config import mxp

from bot.database import db

from bot.queue.globals import user_tasks
from bot.queue.user_task import UserTaskType, create_and_run_user_task

from bot.handlers.utils import (
    get_strings,
    add_handler_routines,
    send_reply,
    thread_pool
)
from bot.handlers.balance import (
    check_if_user_has_enough_tokens,
    show_balance,
    show_balance_if_message_queue_is_full,
    ShowBalanceSource,
)
from bot.handlers.chat_mode import maybe_switch_chat_mode_to_default_because_not_enough_tokens
from bot.handlers.settings import maybe_switch_model_to_default_because_not_enough_tokens
from bot.handlers.tokens import convert_transcribed_seconds_to_bot_tokens
from bot.handlers.constants import NewDialogButtonData


logger = logging.getLogger(__name__)


@add_handler_routines(
    ignore_if_bot_is_not_mentioned=True,
    check_if_previous_message_is_answered=True,
)
async def message_handle(
    update: Update,
    context: CallbackContext,
    message_text: Optional[str] = None,
    use_new_dialog_timeout: bool = True,
):
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    strings = get_strings(user_id)

    message_text = message_text or update.effective_message.text
    db.set_user_attribute(user_id, "last_message_text", message_text)

    # remove bot mention (in group chats)
    if update.effective_chat.type != "private":
        message_text = message_text.replace(config.bot_username, "").strip()

    # switch to default chat mode if user has not enough tokens
    await maybe_switch_chat_mode_to_default_because_not_enough_tokens(
        bot=context.bot,
        user_id=user_id,
        chat_id=chat_id,
    )

    # switch to default model if user has not enough tokens
    await maybe_switch_model_to_default_because_not_enough_tokens(
        bot=context.bot,
        user_id=user_id,
        chat_id=chat_id,
    )

    # define in which queue to process the task
    if not check_if_user_has_enough_tokens(user_id=user_id):
        if config.enable_message_queue:
            user_task_type = UserTaskType.MESSAGE_QUEUE
        else:
            await show_balance(
                bot=context.bot,
                user_id=user_id,
                chat_id=chat_id,
                source=ShowBalanceSource.NOT_ENOUGH_TOKENS,
            )
            return
    else:
        user_task_type = UserTaskType.ASYNCIO_QUEUE

    # message queue is full
    if await show_balance_if_message_queue_is_full(
        bot=context.bot,
        user_id=user_id,
        chat_id=chat_id,
        user_task_type=user_task_type,
    ):
        return

    if not message_text:
        await update.effective_message.reply_text(
            strings["empty_message"],
            parse_mode=ParseMode.HTML
        )
        return

    if await check_if_dialog_timeout_happened(update, context, use_new_dialog_timeout=use_new_dialog_timeout):
        return

    # do subtract tokens?
    if user_task_type == UserTaskType.ASYNCIO_QUEUE:
        do_subtract_tokens = True
    elif user_task_type == UserTaskType.MESSAGE_QUEUE:
        do_subtract_tokens = False
    else:
        raise ValueError(f"Unknown user_task_type: {user_task_type}")

    # mxp
    distinct_id, event_name, properties = (
        user_id,
        "send_message",
        {
            "dialog_id": db.get_user_attribute(user_id, "current_dialog_id"),
            "chat_mode": db.get_user_attribute(user_id, "current_chat_mode"),
            "model": db.get_user_attribute(user_id, "current_model"),
            "user_task_type": user_task_type.name
        }
    )
    mxp.track(distinct_id, event_name, properties)

    await create_and_run_user_task(
        user_task_type=user_task_type,
        bot=context.bot,
        user_id=user_id,
        chat_id=chat_id,
        message_text=message_text,
        do_subtract_tokens=do_subtract_tokens
    )


@add_handler_routines()
async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type != ChatType.PRIVATE:
        return
    user_id = update.effective_user.id
    strings = get_strings(user_id)
    text = strings["edited_message"]
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


@add_handler_routines(check_if_previous_message_is_answered=True)
async def retry_handle(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    strings = get_strings(user_id)

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        text = strings["no_message_to_retry"]
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    last_dialog_message = dialog_messages.pop()
    # last message was removed from the context
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)

    await message_handle(
        update,
        context,
        message_text=last_dialog_message["user"],
        use_new_dialog_timeout=False,
    )


async def check_if_dialog_timeout_happened(update: Update, context: CallbackContext, use_new_dialog_timeout: bool = True):
    user_id = update.effective_user.id
    strings = get_strings(user_id)
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    # handle new dialog timeout case
    ask_new_dialog = False

    if chat_mode != "artist" and use_new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
        last_message_ts = db.get_user_attribute(user_id, "last_message_ts")
        if last_message_ts is not None:  # backward compatibility
            elapsed_seconds = (datetime.now() - last_message_ts).total_seconds()
            if elapsed_seconds > config.new_dialog_timeout:
                ask_new_dialog = True

    if ask_new_dialog:
        buttons = [
            InlineKeyboardButton(strings["new_dialog_button"], callback_data=NewDialogButtonData(True).dump()),
            InlineKeyboardButton(strings["continue_dialog_button"], callback_data=NewDialogButtonData(False).dump()),
        ]
        await update.effective_message.reply_text(
            strings["ask_new_dialog_due_to_timeout"],
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([buttons]),
        )
        return True
    else:
        return False


@add_handler_routines(
    check_if_previous_message_is_answered=True
)
async def new_dialog_timeout_confirm_handle(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    strings = get_strings(user_id)

    use_new_dialog = NewDialogButtonData.load(update.callback_query.data).use_new_dialog

    if use_new_dialog:
        db.start_new_dialog(user_id)
        chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
        chat_mode_name = config.chat_modes[chat_mode]["name"][strings.lang]
        text = (
            strings["starting_new_dialog_due_to_timeout"]
            .format(chat_mode_name=chat_mode_name)
        )
        await send_reply(
            update.effective_message,
            text=text,
            parse_mode=ParseMode.HTML,
            try_edit=True,
        )
    else:
        try:
            await update.effective_message.delete()
        except:
            pass

    message_text = db.get_user_attribute(user_id, "last_message_text")
    await message_handle(
        update,
        context,
        message_text=message_text,
        use_new_dialog_timeout=False
    )


@add_handler_routines(check_if_previous_message_is_answered=True)
async def new_dialog_handle(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    strings = get_strings(user_id)

    db.start_new_dialog(user_id)
    text = strings["new_dialog"]
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    current_model = db.get_user_attribute(user_id, "current_model")

    text = ""
    if config.chat_modes[chat_mode]["model_type"] == "text":
        text += f"<i>{config.models['info'][current_model]['name']}</i>: "

    text += config.chat_modes[chat_mode]["welcome_message"][strings.lang]

    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


@add_handler_routines()
async def cancel_handle(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    strings = get_strings(user_id)

    if user_id in user_tasks:
        user_task = user_tasks[user_id]
        await user_task.cancel()

        text = strings["canceled"]
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)
    else:
        text = strings["nothing_to_cancel"]
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


@add_handler_routines(
    ignore_if_bot_is_not_mentioned=True,
    check_if_previous_message_is_answered=True,
)
async def voice_message_handle(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    strings = get_strings(user_id)

    # voice messages bot ads
    if config.enable_voice_messages_bot_ads:
        text = strings["voice_messages_bot_ads"]
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton(text=strings["voice_messages_bot_ads_button"], url="https://t.me/my_voice_messages_bot?start=source=chatgpt_karfly_bot")],
        ])

        await update.effective_message.reply_text(
            text,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
            parse_mode=ParseMode.HTML
        )

        asyncio.sleep(5.0)

    if not check_if_user_has_enough_tokens(user_id=user_id):
        if config.enable_message_queue:
            show_balance_source = ShowBalanceSource.VOICE_MESSAGE
        else:
            show_balance_source = ShowBalanceSource.NOT_ENOUGH_TOKENS

        await show_balance(
            bot=context.bot,
            user_id=user_id,
            chat_id=chat_id,
            source=show_balance_source,
        )
        return

    voice = update.effective_message.voice
    if voice.duration > 3 * 60:
        text = strings["voice_message_is_too_long"]
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        voice_ogg_path = tmp_dir / "voice.ogg"

        # download
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_ogg_path)

        # convert to mp3
        voice_mp3_path = tmp_dir / "voice.mp3"
        loop = asyncio.get_running_loop()
        def _task_fn(): return pydub.AudioSegment.from_file(voice_ogg_path).export(voice_mp3_path, format="mp3")
        await loop.run_in_executor(thread_pool, _task_fn)

        # transcribe
        with open(voice_mp3_path, "rb") as f:
            transcribed_text = await openai_utils.transcribe_audio(f)

            if transcribed_text is None:
                transcribed_text = ""

    text = f"🎤: <i>{transcribed_text}</i>"
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)

    # token usage
    _n = db.get_user_attribute(user_id, "n_transcribed_seconds")
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + _n)

    n_used_bot_tokens = convert_transcribed_seconds_to_bot_tokens("whisper-1", voice.duration)
    initial_balance = db.get_user_attribute(user_id, "token_balance")

    new_balance = max(0, initial_balance - n_used_bot_tokens)
    db.set_user_attribute(user_id, "token_balance", new_balance)

    # mxp
    distinct_id, event_name, properties = (
        user_id,
        "send_voice_message",
        {"n_used_bot_tokens": n_used_bot_tokens}
    )
    mxp.track(distinct_id, event_name, properties)

    await message_handle(update, context, message_text=transcribed_text)
