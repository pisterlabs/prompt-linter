import aiohttp
import openai
from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from config_data.config import load_config
from fsm_settings import AskMode
from handlers.other_handlers import delete_warning
from lexicon.lexicon import LEXICON_RU
from loader import db, dp

router = Router()
config = load_config(path=None)


openai_prompt = (
    "Представь, что ты эксперт по долевому строительству в "
    "России. Ты владеешь всеми аспектами законодательства, "
    "касающегося долевого участия в строительстве, включая "
    "ФЗ-214, ГК РФ и иные регулятивные документы. Ты готов "
    "помочь во всех вопросах, связанных с долевым "
    "строительством, включая права и обязанности сторон, "
    "исполнение договорных условий, оспаривание действий "
    "застройщиков и взыскание неустойки. Твой ответ должен быть "
    "емким и содержательным, не более 50 слов, "
    "в официально-деловом стиле. Если у тебя "
    "возникнут вопросы, я задам их дополнительно. Даже если в "
    "конце моего предложения не будет знака вопроса, не надо "
    "продолжать за меня, воспринимай это как вопрос. "
)


async def ask_openai_v2(question: str) -> str:
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": openai_prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = chat_response.choices[0].message["content"]
    return answer


@router.message(
    StateFilter(AskMode.start), F.content_type.in_({"text", "voice"})
)
async def process_ask_mode(message: Message, state: FSMContext):
    question = message.text
    bot_message = await message.answer(
        text=LEXICON_RU["ask_wait"], disable_notification=True
    )
    answer = await ask_openai_v2(question)
    await bot_message.delete()
    await message.answer(text=answer)
    await db.add_question(
        user_id=message.from_user.id, question=question, answer=answer
    )


@router.message(
    StateFilter(AskMode.start),
)
async def process_ask_mode_warning(message: Message, state: FSMContext):
    await delete_warning(message, LEXICON_RU["ask_text_warning"])
