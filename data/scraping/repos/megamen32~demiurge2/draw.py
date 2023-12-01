import json
import random
import re
import time
import traceback

import langdetect
import openai
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton

import config
import tgbot
from imaginepy import Imagine, AsyncImagine

from imaginepy import Imagine, Style, Ratio
import asyncio
import io
from aiogram import types

from config import dp, bot
from datebase import Prompt, ImageMidjourney
from gpt import gpt_acreate
from tgbot import get_chat_data, get_storage_from_chat, dialog_append, dialog_append_raw

MIDJOURNEY = 'MIDJOURNEY'
UNSTABILITY = 'UNSTABILITY'

imagine = None


async def gen_img(prompt, ratio, style):
    if (isinstance(style, Style) )and 'style' not in prompt:
        if style==Style.NO_STYLE:
            style=None
        else:
            prompt+=f". {style.name.lower().replace('_',' ').replace(' v2','')} style"
            style=MIDJOURNEY


    if style == UNSTABILITY:
        from imagine import agenerate_image_stability

        imd_data = await agenerate_image_stability(prompt, style)
        return imd_data[0], None,style
    else:# style == MIDJOURNEY:
        if style!=MIDJOURNEY and isinstance(style,str) :
            if 'style' not in prompt:
                prompt += f". {style.lower().replace('_', ' ').replace(' v2','')} style"
            style = MIDJOURNEY
        from imagine import generate_image_midjourney
        ratio_str = ratio.name.lower().replace('ratio_', '').replace('x',':')

        # Подготовка сообщения для Midjourney
        prompt += f' --ar {ratio_str}'
        img_data, img_url = await generate_image_midjourney(prompt)

        return img_data, img_url,style


async def upscale_image_imagine(img_data):
    global imagine
    if imagine is None:
        imagine = AsyncImagine()
    img_data = await imagine.upscale(image=img_data)
    return img_data


async def improve_prompt(prompt, storage_id,user_id):
    # Detect the language of the prompt
    try:
        lang = langdetect.detect(prompt)
    except langdetect.lang_detect_exception.LangDetectException:
        lang = 'en'

    # If the language is not English, translate and improve it
    if lang == 'ru' or lang == 'uk' or lang == 'mk':
        user_data = await dp.storage.get_data(chat=storage_id)
        history = user_data.get('history', [])
        chat_response = await gpt_acreate(
            model="gpt-3.5-turbo",
            messages=history + [
                {"role": "system",
                 "content": '''Use the following info as a reference to create ideal Midjourney prompts.

•	Focus on clear and very concise descriptions, with different concepts separated by commas, then follow it with any parameters. Parameters are not separated by commas.
•	Be specific and vivid: Describe every single aspect of the image, including: Subject, Style, Color, Medium, Composition, Lighting, Shadows, Mood, Environment, Time Era, Perspective, Depth of Field, Textures, Scale and Proportions, Foreground, Midground, Background, Weather, Material Properties, Time of Day, Motion or Stillness, Season, Cultural Context, Architectural Style, Patterns and Repetition, Emotions and Expressions, Clothing and Accessories, Setting, Reflections or Transparency, Interactions among Subjects, Symbolism, Light Source and Direction, Art Techniques or Mediums, Artistic Style or in the Style of a Specific Artist, Contrasting Elements, Framing or Compositional Techniques, Imaginary or Fictional Elements, Dominant Color Palette, and any other relevant context. 

•	Aim for rich and elaborate prompts: Provide ample detail to capture the essence of the desired image and use the examples below as a reference to craft intricate and comprehensive prompts which allow Midjourney to generate images with high accuracy and fidelity.
•	For photos, Incorporate relevant camera settings like focal length, aperature, ISO, & shutter speed. Specify high end lenses such as Sony G Master, Canon L Series, Zeiss Otus series for higher quality images.
•	Select the aspect ratio by adding the --ar <value>:<value> parameter. Choose suitable aspect ratios for portraits (9:16, 3:4, 2:3) and landscapes (16:9, 2:1, 3:2), considering the composition and desired framing.
•	Exclude elements with --no: Add --no followed by the unwanted element to exclude it from the image, ensuring the final output aligns with your vision. Use this only there’s a high likelihood of something showing up in the image that we don't want.
•	Diversify your prompts: Explore various styles, moods, colors, art mediums, and aspect ratios to create a wide range of visually appealing and unique images.

Here are 2 example prompts. The first is artistic, the last is photo. Use these examples to determine desired length of each prompt.

•	Digital art of an enchanting piano recital set within a serene forest clearing, a grand piano as the centerpiece, the musician, a young woman with flowing locks and an elegant gown, gracefully playing amidst the vibrant green foliage and deep brown tree trunks, her fingers dancing across the keys with an air of passion and skill, soft pastel colors adding a touch of whimsy, warm, dappled sunlight filtering through the leaves, casting a dreamlike glow on the scene, a harmonious fusion of music and nature, eye-level perspective immersing the viewer in the tranquil woodland setting, a captivating blend of art and the natural world --ar 2:1•	Detailed charcoal drawing of a gentle elderly woman, with soft and intricate shading in her wrinkled face, capturing the weathered beauty of a long and fulfilling life. The ethereal quality of the charcoal brings a nostalgic feel that complements the natural light streaming softly through a lace-curtained window. In the background, the texture of the vintage furniture provides an intricate carpet of detail, with a monochromatic palette serving to emphasize the subject of the piece. This charcoal drawing imparts a sense of tranquillity and wisdom with an authenticity that captures the subject's essence.
•	Astounding astrophotography image of the Milky Way over Stonehenge, emphasizing the human connection to the cosmos across time. The enigmatic stone structure stands in stark silhouette with the awe-inspiring night sky, showcasing the complexity and beauty of our galaxy. The contrast accentuates the weathered surfaces of the stones, highlighting their intricate play of light and shadow. Sigma Art 14mm f/1.8, ISO 3200, f/1.8, 15s --ar 16:9 

You will receive a text prompt and then create one creative prompt for the Midjourney AI art generator using the best practices mentioned above. Do not include explanations in your response. List one prompt on English language with correct syntax without unnecessary words. Promt is: ''' + prompt}
            ],
            max_tokens=200,user_id=user_id
        )

        # Extract the model's response
        improved_prompt = chat_response['choices'][0]['message']['content']
        # Удаление символов кавычек
        cleaned_text = improved_prompt.replace('"', '').replace("'", '').replace('translates to', '')

        # Поиск английского текста с использованием регулярного выражения
        improved_prompt = ' '.join(re.findall(r'\b[A-Za-z]+\b', cleaned_text))
        if improved_prompt.startswith('draw'):
            improved_prompt = improved_prompt.replace('draw', '', 1)
        fak = f'Improved image generation prompt from "{prompt}" to "{improved_prompt}. And starts drawing."'
        await dialog_append_raw(storage_id, fak, role='system')

        # Remove the model's name from the response
        improved_prompt = re.sub(r'^.*?:', '', improved_prompt).strip()

        return improved_prompt

    # If the language is English, return the original prompt
    return prompt

width = 3
raws = 6

# Сколько страниц стилей доступно
PAGE_SIZE = width * raws
def create_style_keyboard(prompt, start_index=0):
    styles = list(Style.__members__.keys())
    ratios = list(Ratio.__members__.keys())
    prompt_db, _ = Prompt.get_or_create(text=prompt)
    kb = types.InlineKeyboardMarkup(resize_keyboard=True)

    pages = len(styles) // (PAGE_SIZE)
    use_pages=False
    # Выводимые стили в зависимости от текущей страницы (start_index)
    horizontal_styles = styles[start_index * width * raws:(start_index + 1) * width * raws]

    for i in range(raws):
        # Добавление горизонтального ряда кнопок со случайными стилями
        buttons = [
            types.InlineKeyboardButton(style.lower(), callback_data=f'style_{prompt_db.id}_{style}')
            for style in horizontal_styles[i * width:(i + 1) * width]
        ]
        kb.row(*buttons)

    if use_pages:

        # Добавить кнопки "Вперед" и "Назад", если это необходимо
        if start_index > 0:
            kb.add(types.InlineKeyboardButton('<<', callback_data=f'prev_{prompt_db.id}_{start_index}'))
        if start_index  < len(styles)//PAGE_SIZE:
            kb.add(types.InlineKeyboardButton('>>', callback_data=f'next_{prompt_db.id}_{start_index}'))
    else:
        # Используем вариант со списком страниц
        kb.row(*[types.InlineKeyboardButton(f"{i + 1 }" if i!=start_index else f">{i+1}<", callback_data=f'page_{prompt_db.id}_{i+1}') for i in
                 range(pages+ 1) ])

    buttons = [
        types.InlineKeyboardButton(ratio.lower().replace('ratio_', ''), callback_data=f'ratio_{prompt_db.id}_{ratio}')
        for ratio in ratios
    ]
    kb.row(*buttons)

    buttons = []
    buttons.append(types.InlineKeyboardButton(MIDJOURNEY, callback_data=(f'style_{prompt_db.id}_{MIDJOURNEY}')))
    buttons.append(types.InlineKeyboardButton(UNSTABILITY, callback_data=(f'style_{prompt_db.id}_{UNSTABILITY}')))
    kb.row(*buttons)

    return kb


@dp.callback_query_handler(lambda callback: callback.data.startswith('ratio')or callback.data.startswith('page') or callback.data.startswith('style') or callback.data.startswith('prev') or callback.data.startswith('next'))
async def handle_ratio_callback(query: types.CallbackQuery):
    # Обработка callback для соотношений
    user_data, chat_id = await get_chat_data(query.message)
    command, id, text = query.data.split('_', 2)
    prompt = Prompt.get_by_id(id).text
    redraw = True
    if text in Style.__members__ or text in [MIDJOURNEY, UNSTABILITY]:
        user_data['style'] = text
        await query.answer(f"Set style to {text}.")
    elif text in Ratio.__members__:
        user_data['ratio'] = text
        await query.answer(f"Set ratio to {text}.")
    elif command == "prev":
        # Decrease the start_index
        user_data['style_start_index'] = max(0, user_data.get('style_start_index', 0) - 1)
        await query.answer("Scrolling to previous styles.")
        redraw = False
    elif command == "next":
        # Increase the start_index
        user_data['style_start_index'] = min((len(Style.__members__) - 1)//PAGE_SIZE, user_data.get('style_start_index', 0) + 1)
        await query.answer("Scrolling to next styles.")
        redraw = False
    elif command == "page":
        # Set the start_index to the selected page
        user_data['style_start_index'] = (int(text) - 1)
        await query.answer(f"Set page to {text}.")
        redraw = False
    else:
        await query.answer("Unknown option.")
    await dp.storage.set_data(chat=chat_id, data=user_data)
    if not redraw:
        kb = create_style_keyboard(prompt,user_data.get('style_start_index',0))  # Update keyboard with new styles
        await bot.edit_message_reply_markup(chat_id=query.message.chat.id,
                                            message_id=query.message.message_id,
                                            reply_markup=kb)
    else:
        await draw_and_answer(prompt, query.message.chat.id, query.message.message_thread_id,query.from_user.id)


def translate_promt(prompt):
    from translate import Translator
    translator = Translator(from_lang='ru', to_lang="en")
    translation = translator.translate(prompt)
    return translation


async def progress_bar(text, msg:types.Message, timeout=60, cancel: asyncio.Event = None):
    bar_length = 10
    sleep_time = max(10,timeout // bar_length)
    last_typing_time = 0
    emoji_sets = [  # Массив массивов эмодзи
        ["🟩", "🟨", "🟧", "🟦", "🟪", "🟥"],
        ["⭐️", "🌟", "🤩", "💫", "✨", "🌠"],
        ["❤️", "🧡", "💛", "💚", "💙", "💜"],
        ["🟠", "🟡", "🟢", "🔵", "🟣", "🔴"],
    ]

    bar_emoji = random.choice(emoji_sets)  # Выбираем набор эмодзи случайным образом
    sybmov = random.choice(['⬜️', '  '])
    for i in range(bar_length):
        progress = (i % bar_length) + 1

        bar_str = [sybmov] * bar_length
        bar_str[:progress] = [bar_emoji[i // 2] for _ in range(progress)]  # меняем цвет бара по мере выполнения задачи

        current_time = time.time()
        if current_time - last_typing_time >= 5:  # Проверяем, прошло ли 5 секунд с последнего отправления "typing"
            await bot.send_chat_action(chat_id=msg.chat.id,message_thread_id=msg.message_thread_id, action='TYPING')
            last_typing_time = current_time  # Обновляем время последнего отправления "typing"

        await asyncio.sleep(sleep_time)
        if cancel and cancel.is_set():  # Проверяем, установлен ли флаг отмены
            break
        await bot.edit_message_text(chat_id=msg.chat.id,message_id=msg.message_id,text=f'{text}\n' + ''.join(bar_str),ignore=True)
async def draw_and_answer(prompt, chat_id, reply_to_id,user_id):
    user_data, user_id = await get_storage_from_chat(chat_id, reply_to_id)
    ratio = Ratio[user_data.get('ratio', 'RATIO_4X3')]
    try:
        style = Style[user_data.get('style', 'ANIME_V2')]
    except:
        style = user_data.get('style', 'ANIME_V2')
    msg = await bot.send_message(chat_id=chat_id, text=f"Creating image... {style}\n{ratio} \n{prompt}",
                                 reply_to_message_id=reply_to_id,ignore=True)
    error = False
    cancel_event = asyncio.Event()
    try:
        if re.match('[а-яА-Я]+', prompt):
            prompt = translate_promt(prompt)
        if config.USE_API:
            moderate = await openai.Moderation.acreate(prompt)
            is_sexual = moderate['results'][0]['categories']['sexual']
        else:
            is_sexual = False
        if is_sexual:
            style = UNSTABILITY
        else:
            prompt = await improve_prompt(prompt, chat_id,user_id)

        new_text = f"Finishing image... {style}\n{ratio} \n{prompt}"

        asyncio.create_task(progress_bar(new_text,msg,cancel=cancel_event))
        old_style=style
        img_file, url,style = await gen_img(prompt, ratio, style)
        if img_file is None:
            raise Exception("500 server image generator error ")


        photo = None
        start_index=user_data.get('style_start_index', 0)
        kb :types.InlineKeyboardMarkup= create_style_keyboard(prompt,start_index)
        if False and isinstance(style, Style):
            photo = await bot.send_photo(chat_id=chat_id, photo=io.BytesIO(img_file), caption=f'{prompt}',
                                         reply_to_message_id=reply_to_id)
            img_file = await upscale_image_imagine(img_file)
        else :
            img_db = ImageMidjourney.create(prompt=prompt, url=url)

            btns = [InlineKeyboardButton(text=f"U {_ + 1}", callback_data=f"imagine_{_ + 1}_{img_db.id}") for _ in
                    range(4)]
            kb.row(*btns[:2])
            kb.row(*btns[-2:])
        photo2 = await bot.send_photo(chat_id=chat_id, photo=io.BytesIO(img_file),
                                      caption=f'{prompt}\n{old_style}\n{ratio}', reply_markup=kb,
                                      reply_to_message_id=reply_to_id)
        if photo is not None:
            await photo.delete()
        if not url:
            file_info = await bot.get_file(photo2.photo[-1].file_id)
            url = f"https://api.telegram.org/file/bot{config.TELEGRAM_BOT_TOKEN}/{file_info.file_path}"

        di= {'prompt': prompt, 'style': style.name if isinstance(style,Style) else style, 'image generated without exception':True}
        await tgbot.dialog_append(photo2,json.dumps(di,ensure_ascii=False),'function',name='draw')
    except Exception as e:
        traceback.print_exc()
        await bot.send_message(chat_id=chat_id,text= f"An error occurred while generating the image. {e}",reply_to_message_id=reply_to_id)
        di= {'prompt': prompt, 'style': style.name  if isinstance(style,Style) else style, 'image generated without exception':traceback.format_exc(0,False)}
        await tgbot.dialog_append(msg, json.dumps(di, ensure_ascii=False), 'function', name='draw')
    finally:
        cancel_event.set()
        await bot.delete_message(msg.chat.id, msg.message_id,thread_id=msg.message_thread_id)


@dp.message_handler(commands=['draw'])
async def handle_draw(message: types.Message):
    prompt = message.get_args()
    if not prompt:
        await message.reply("Please provide a description for the image.")
        return

    return await draw_and_answer(prompt, message.chat.id, message.message_thread_id,message.from_user.id)


def create_settings_keyboard():
    styles = list(Style.__members__.keys()) + [MIDJOURNEY, UNSTABILITY]
    ratios = list(Ratio.__members__.keys())
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)

    # Add ratio buttons
    ratio_buttons = [types.KeyboardButton(ratio) for ratio in ratios]
    keyboard.row(*ratio_buttons)

    # Add a separator
    keyboard.row(types.KeyboardButton("-" * 10))

    # Add style buttons in groups of 5
    for i in range(0, len(styles), 5):
        style_buttons = [types.KeyboardButton(style) for style in styles[i:i + 5]]
        keyboard.row(*style_buttons)

    return keyboard


class DrawingSettings(StatesGroup):
    settings = State()


@dp.message_handler(commands=['draw_settings'])
async def handle_draw_settings(message: types.Message, state: FSMContext):
    keyboard = create_settings_keyboard()
    await DrawingSettings.settings.set()
    user_data, chat_id = await get_chat_data(message)
    style = user_data.get('style', 'ANIME_V2')
    if style in Style.__members__:
        style = Style[style]
    ratio = user_data.get('ratio', 'RATIO_4X3')
    if ratio in Ratio.__members__:
        ratio = Ratio[ratio]

    await message.reply(f"Please choose style and ratio for your drawings.{style} {ratio}", reply_markup=keyboard)


@dp.message_handler(state=DrawingSettings.settings.state)
async def handle_style_and_ratio(message: types.Message, state: FSMContext):
    user_data, chat_id = await get_chat_data(message)
    text = message.text
    if text in Style.__members__:
        user_data['style'] = text
        await message.reply(f"Set style to {text}.")
    elif text in Ratio.__members__:
        user_data['ratio'] = text
        await message.reply(f"Set ratio to {text}.")
    else:
        await message.reply("Unknown option.")
    await state.finish()
    await dp.storage.set_data(chat=chat_id, data=user_data)



