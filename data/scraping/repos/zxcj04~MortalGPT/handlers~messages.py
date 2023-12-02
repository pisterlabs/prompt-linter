import asyncio
import logging
import os

from telegram import File, Update
from telegram.ext import ContextTypes
from openai.error import OpenAIError
from pydub import AudioSegment

from lib import gpt, config, errorCatch, constants, user_store


MESSAGE_LOCKS = {}


async def editMsg(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id,
    message_id,
    text: str,
    reply_markup=None,
):
    text = text.strip()

    if len(text) == 0:
        return

    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
        )
    except Exception as e:
        errorCatch.logError(e)
        pass


async def updateTask(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    targetFn,
    initMsg: str,
    *args,
    **kwargs,
):
    global MESSAGE_LOCKS

    user_id = update.effective_user.id

    if user_id not in MESSAGE_LOCKS:
        MESSAGE_LOCKS[user_id] = asyncio.Lock()

    async with MESSAGE_LOCKS[user_id]:
        chat_id = update.effective_chat.id
        id = await context.bot.send_message(
            chat_id=chat_id,
            text=initMsg,
            disable_notification=True,
        )

        await targetFn(update, context, id.message_id, *args, **kwargs)


def chatSentencesGenerator(word_generator):
    sentence = ""

    for word in word_generator:
        sentence = constants.CC.convert(sentence + word)

        try:
            is_punctuation = (
                any([p in word for p in constants.PUNCTUATIONS])
                or word in constants.PUNCTUATIONS
            )
        except:
            is_punctuation = False

        if is_punctuation and len(sentence) > 10:
            yield sentence
            sentence = ""

    yield sentence


async def updateChatToUser(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message_id,
    isRetry=False,
):
    user_id = update.effective_user.id
    user_name = update.effective_user.name
    chat_id = update.effective_chat.id
    chat_text = update.effective_message.text
    forward_message = True

    user_store.STORE.set_user_name(user_id, user_name)

    if isRetry:
        logging.info(f"User {user_name} is retrying")
        await errorCatch.sendMessageToAdmin(update, context, "> RETRY <")

        _, chat_text = user_store.STORE.pop_to_last_user_message(user_id)

        if chat_text is None:
            try:
                await editMsg(context, chat_id, message_id, "﹝不知道要怎麼回答﹞")
                return
            except Exception as e:
                errorCatch.logError(e)
                pass

        forward_message = False

    await errorCatch.sendMessageToAdmin(
        update, context, forward_message=forward_message
    )

    paragraph = ""
    last_paragraph = ""
    is_code_block = False
    is_new_paragraph = False
    last_message_id = message_id

    try:
        formated_chat_text = chat_text.replace("\n", "\\n")
        logging.info(f"User {user_name} ask: {formated_chat_text}")

        answer_generator = gpt.get_answer(user_id, chat_text)
        sentencesGenerator = chatSentencesGenerator(answer_generator)

        for sentence in sentencesGenerator:
            if len(sentence) == 0:
                continue

            if is_new_paragraph:
                id = await context.bot.send_message(
                    chat_id=chat_id,
                    text="﹝正在思考﹞",
                    disable_notification=True,
                )
                last_message_id = message_id
                message_id = id.message_id

                is_new_paragraph = False

            last_is_code_block = is_code_block

            for c in sentence:
                if c == "`":
                    is_code_block = not is_code_block

            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action="typing",
                )

                if "\n" in sentence and not is_code_block:
                    pre_sentence, post_sentence = sentence.rsplit("\n", 1)
                    if "```" in post_sentence and last_is_code_block:
                        p, t = post_sentence.split("```", 1)
                        pre_sentence += p + "\n```"
                        post_sentence = t

                    paragraph += pre_sentence
                    if len(paragraph) != 0 and len(pre_sentence) != 0:
                        await editMsg(context, chat_id, message_id, paragraph)
                    user_store.STORE.add_assistant_message(user_id, paragraph)

                    if len(paragraph.strip()) > 0:
                        last_paragraph = paragraph
                        is_new_paragraph = True

                    paragraph = post_sentence
                else:
                    paragraph += sentence
                    await editMsg(context, chat_id, message_id, paragraph)

            except Exception as e:
                errorCatch.logError(e)
                pass

    except OpenAIError as e:
        await errorCatch.sendMessageToAdmin(
            update, context, f"{e.__str__()}: {e.user_message}"
        )
        await errorCatch.sendTryAgainError(update, context, e.user_message)
        return
    except Exception as e:
        errorCatch.logError(e)
        await errorCatch.sendMessageToAdmin(update, context, f"{e.__str__()}")
        await errorCatch.sendErrorMessage(update, context)
        return

    try:
        if len(paragraph) == 0:
            message_id = last_message_id
            paragraph = last_paragraph

        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=constants.INLINE_KEYBOARD_MARKUP_RETRY_RESET,
            text=paragraph,
        )
    except Exception as e:
        errorCatch.logError(e)
        pass

    user_store.STORE.add_assistant_message(user_id, paragraph)
    logging.info(f"Answered to User {user_name}")


async def chat(
    update: Update, context: ContextTypes.DEFAULT_TYPE, isRetry=False
):
    asyncio.get_event_loop().create_task(
        updateTask(
            update,
            context,
            targetFn=updateChatToUser,
            initMsg="﹝正在思考﹞",
            isRetry=isRetry,
        )
    )


async def normalChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await chat(update, context)


async def retryChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await chat(update, context, isRetry=True)


async def updateWhisperChatToUser(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message_id,
    tg_file: File,
):
    user_name = update.effective_user.name
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    is_long = False

    try:
        logging.info(f"User {user_name} ask for whisper api")

        file_name = f"tmp/{user_id}.mp3"
        await tg_file.download_to_drive(file_name)
        audio_file: AudioSegment = AudioSegment.from_file(file_name)
        if audio_file.duration_seconds > 30 and str(user_id) != str(
            config.ADMIN_ID
        ):
            audio_file: AudioSegment = AudioSegment.from_file(
                file_name, duration=30
            )
            is_long = True
        audio_file.export(file_name, format="mp3")

        with open(file_name, "rb") as f:
            answer = gpt.get_whisper_api_answer(f)

        if len(answer) == 0:
            await editMsg(
                context,
                chat_id,
                message_id,
                "﹝無法辨識﹞",
                reply_markup=constants.INLINE_KEYBOARD_MARKUP_RETRY_RESET,
            )
            return

        answers = [answer[i : i + 500] for i in range(0, len(answer), 500)]
        is_first = True

        if is_long:
            await editMsg(
                context=context,
                chat_id=chat_id,
                message_id=message_id,
                text="﹝音訊超過 30 秒，只會回傳前 30 秒的文字﹞",
            )
            is_first = False

        user_store.STORE.add_user_message(user_id, "使用 whisper api 聽寫這段音訊")

        for index, answer in enumerate(answers):
            reply_markup = (
                None
                if index != len(answers) - 1
                else constants.INLINE_KEYBOARD_MARKUP_RESET
            )
            if is_first:
                logging.info("Send whisper api answer")
                await editMsg(
                    context=context,
                    chat_id=chat_id,
                    message_id=message_id,
                    text=answer,
                    reply_markup=reply_markup,
                )
                is_first = False
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=answer,
                    reply_markup=reply_markup,
                )

            user_store.STORE.add_assistant_message(user_id, answer)
    except Exception as e:
        errorCatch.logError(e)
        await errorCatch.sendMessageToAdmin(update, context, f"{e.__str__()}")
        await errorCatch.sendErrorMessage(update, context)

    logging.info(f"Answered to User {user_name}")


async def whisperChat(
    update: Update, context: ContextTypes.DEFAULT_TYPE, tg_file: File
):
    asyncio.get_event_loop().create_task(
        updateTask(
            update,
            context,
            targetFn=updateWhisperChatToUser,
            initMsg="﹝正在聽寫﹞",
            tg_file=tg_file,
        )
    )


async def videoChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_file: File = await update.message.video.get_file()
    await whisperChat(update, context, tg_file=tg_file)


async def audioChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_file: File = await update.message.audio.get_file()
    await whisperChat(update, context, tg_file=tg_file)


async def fileChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_file: File = await update.message.document.get_file()
    await whisperChat(update, context, tg_file=tg_file)


async def voiceChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_file: File = await update.message.voice.get_file()
    await whisperChat(update, context, tg_file=tg_file)


async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == constants.CallBackType.RESET:
        await errorCatch.sendMessageToAdmin(update, context, "> RESET <")
        logging.info(f"User {update.effective_user.name} is resetting")
        await config.resetUser(
            context, query.from_user.id, query.message.chat_id
        )
    elif query.data == constants.CallBackType.DONE:
        user_id = query.from_user.id
        if user_id not in MESSAGE_LOCKS:
            MESSAGE_LOCKS[user_id] = asyncio.Lock()

        async with MESSAGE_LOCKS[user_id]:
            await context.bot.send_sticker(
                chat_id=query.message.chat_id,
                sticker=constants.DONE_STICKER,
            )
    elif query.data == constants.CallBackType.RETRY:
        await retryChat(update, context)


async def chatOtherFallback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text="﹝不知道要怎麼回答﹞",
    )
