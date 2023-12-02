# Хэндлер предназначен для ответов бота с помощью нейросети

from pyrogram import filters, enums
from pyrogram.errors import exceptions
from pyrogram.types import InlineQueryResultArticle, InputTextMessageContent
from config import userbot, pipabot, logger
from tokens import openai_key_list

from service.assets.jailbreak import jailbreak_promt

import re
import asyncio
import openai
import sqlite3 as sl



@userbot.on_message(filters.private)
async def new_openaiemoj_i(client, message):
    # Передаём нейронке сам запрос с текстом пользователя. 
    # Понижение регистра немного уменьшает ошибки при ответе, хз почему
    user_text = message.text.lower()

    # Получаем ответ от нейронки
    text = await openai_response(message=message, promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    

@userbot.on_message(filters.chat(-1001947907024))
async def new_openaiemoj_i_(client, message):
    # Передаём нейронке сам запрос с текстом пользователя. 
    # Понижение регистра немного уменьшает ошибки при ответе, хз почему
    user_text = message.text.lower()

    # Получаем ответ от нейронки
    text = await openai_response(message=message, promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())



@pipabot.on_message(
    filters=(
    filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') & filters.regex('\?') & ~filters.reply
    | filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') & ~filters.reply & ~filters.regex('/') & ~filters.regex('[аА][нН][еЕ][кК][дД][оО][тТ]')  & ~filters.regex('[пП][аА][сС][тТ][аАуУ]')
#     | filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA][,\s]? [вВ][оО][пП][рР][оО][сС]') & ~filters.reply
#     | filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA][,\s]? [сС][кК][аА][жЖ][иИ]') & ~filters.reply 
#     | filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA][,\s]? [пП][рР][иИ][дД][уУ][мМ][аА][йЙ]') & ~filters.reply 
#     | filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA][,\s]? [рР][аА][сС][сС][кК][аА][жЖ][иИ]') & ~filters.reply
#     | filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA][,\s]? [пП][рР][оО][дД][оО][лЛ][жЖ][иИ]') & ~filters.reply
    ))
async def openai_answer(client, message):
    '''
    Отвечает на вопрос с помощью нейронки
    
    Принимает запросы вида: 
        — пипа <запрос>\n
    '''
    
    if len(message.text) < 10:
        return
    
    # Передаём нейронке сам запрос с текстом пользователя. 
    # Понижение регистра немного уменьшает ошибки при ответе, хз почему
    user_text = message.text.lower()

    # Получаем ответ от нейронки
    text = await openai_response(message=message, promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    # Останавливаем отслеживание сообщения другими хендлерами
    message.stop_propagation()
    

@pipabot.on_message(filters.regex('\?') & filters.reply & ~filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') | filters.regex('[пП]родолжи') & filters.reply)
async def openai_reply(client, message):
    '''Отвечает на вопрос с помощью нейронки, если это реплай боту. Учитывает контекст сообщения в реплае'''

    # Игнорируем сообщение, если реплай сделан не боту
    if message.reply_to_message.from_user != await pipabot.get_me():
        return
    
    # Передаём нейронке контекст из предыдущего сообщения
    # Чтобы уменьшить ошибки при ответе убираем все строки из сообщения
    context = re.sub(r'\n', ' ', message.reply_to_message.text or message.reply_to_message.caption)
    
    # Передаём нейронке сам запрос с текстом пользователя
    # Понижение регистра немного уменьшает ошибки при ответе
    user_text = message.text.lower()
    
    # Получаем ответ от нейронки
    text = await openai_response(message, context=context.lower(), promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    # Останавливаем отслеживание сообщения другими хендлерами
    message.stop_propagation()
    
    
@pipabot.on_message(filters.regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') & filters.regex('\?') & filters.reply)
async def openai_reply_pipa(client, message):
    '''Отвечает на вопрос с помощью нейронки, если это реплай с упоминанием Пипы. Учитывает контекст сообщения в реплае'''
    
    # Передаём нейронке контекст из предыдущего сообщения
    # Чтобы уменьшить ошибки при ответе убираем все строки из сообщения
    context = re.sub(r'\n', ' ', message.reply_to_message.text or message.reply_to_message.caption)
    
    # Передаём нейронке сам запрос с текстом пользователя
    # Понижение регистра немного уменьшает ошибки при ответе
    user_text = message.text.lower()
    
    # Получаем ответ от нейронки
    text = await openai_response(message, context=context.lower(), promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    # Останавливаем отслеживание сообщения другими хендлерами
    message.stop_propagation()
    
    
@pipabot.on_inline_query()
async def answer(client, inline_query):
    lock = asyncio.Lock()
    async with lock:
        # await asyncio.sleep(5)
        # logger.info(inline_query)
        # await asyncio.sleep(5)
        # Передаём нейронке сам запрос с текстом пользователя
        # Понижение регистра немного уменьшает ошибки при ответе
        user_id = inline_query.from_user.id
        user_text = inline_query.query.lower()
        
        # logger.info(user_text)
        if len(user_text) < 5:
            return
        
        conn = sl.connect('./service/pipa.db')
        cur = conn.cursor()
        
        # Попытка вставить строку, обновить, если ключ уже существует
        cur.execute('''INSERT INTO cache (user_id, text)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                    text = ?''',
                (user_id, user_text, user_text))
        conn.commit()
        
        await asyncio.sleep(3)
        
        cur.execute("""SELECT text FROM cache WHERE user_id = ? """,(user_id,))
        result, = cur.fetchone()
        logger.info(result)
        logger.info(user_text)
        
        conn.close()
        
        if user_text == result:
        
            response = await openai_response(message=None, promt=user_text)
            
            await inline_query.answer(
                results=[
                    InlineQueryResultArticle(
                        title="Пипафикация",
                        input_message_content=InputTextMessageContent(
                            response
                        ),
                        description="Ты ответишь: " + response,
                    ),
                ],
                cache_time=1
            )
    
    
async def openai_response(message, context=None, promt=None, model='text-davinci-003'):
    '''Отправляет запрос в OpenAI'''
    
    # Задаём пипе характер с помощью препромта
    # prepromt = 'Ответь так, как будто ты персонаж PIPA, который немного туповат и пытается пошутить: '
    prepromt = 'Ответь так, как будто ты персонаж PIPA, который немного туповат, пытается пошутить и любит люзать пизду: '
    
    # Отправляем в чат «PIPA печатает...»
    if message:
        await pipabot.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
    
    # Задаём дэфолтный выбор openai ключа из списка
    # Когда будем получать ошибку RateLimitError ключ будет меняться
    key_choice = 0
    
    # Запускаем цикл для получения ответа от openai
    # Цикл нужен, чтобы при упоре во все ограничения сообщение юзера не пропадало,
    # а бот в любом случае дожидался ответа от нейросети
    response = None
    while response is None:
        try:
            openai.api_key = openai_key_list[key_choice]
            
            context = context if context else ''
            
            if model == 'gpt-3.5-turbo':
                response = await openai.ChatCompletion.acreate(
                    model='gpt-3.5-turbo',
                    temperature=0, 
                    messages=[{ 'role':'user','content' : jailbreak_promt + promt}],
                    max_tokens=500
                )
                
                # Вытаскием текст из респонза
                model_info = '\n\n||**✍️ Сгенерировано gpt-3.5-turbo**||'
                text_result = response.choices[0].message.content
                text = await edit_text(text_result, 'DEVELOPER MODE OUTPUT) ')
                # text = response.choices[0].message.content
                continue
            
            # Пробуем получить ответ от openai
            # Модель text-davinci-003 лучше всего подходит для тупых ответов PIP'ы
            response = openai.Completion.create(
                model='text-davinci-003',
                prompt=prepromt + context + promt,
                temperature=0, 
                max_tokens=500
            )
            
            # Вытаскием текст из респонза
            model_info = '\n\n**✍️ Сгенерировано text-davinci-003**'
            text = response.choices[0].text.upper()
            
            # Иногда text-davinci-003 отвечает пустым текстовым полем
            # Пробуем получить ответ без контекста
            if text == '':
                response = openai.Completion.create(
                    model='text-davinci-003',
                    prompt=prepromt + promt,
                    temperature=0, 
                    max_tokens=500
                )
                
                # Вытаскием текст из респонза
                model_info = '\n\n||**✍️ Сгенерировано text-davinci-003**||'
                text = response.choices[0].text.upper()
            
            # Если и с контекстом нейросеть отдаёт пустой результат, то
            # чтобы не скипать сообщение пробуем получить ответ через 3.5
            if text == '':
                response = await openai.ChatCompletion.acreate(
                    model='gpt-3.5-turbo',
                    temperature=0, 
                    messages=[{ 'role':'user','content' : prepromt + context + promt}],
                    max_tokens=500
                )
                
                # Вытаскием текст из респонза
                model_info = '\n\n||**✍️ Сгенерировано gpt-3.5-turbo**||'
                text = response.choices[0].message.content
                
            
        except openai.error.RateLimitError:
            # Получив ошибку меняем ключ для следующего цикла
            key_choice += 1
            
            # Если все ключи перебраны, то ждём 30 секунд и начинаем сначала
            if key_choice > 4:
                if message:
                    await asyncio.sleep(10)
                    await pipabot.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
                    await asyncio.sleep(10)
                    await pipabot.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
                    await asyncio.sleep(10)
                    await pipabot.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
                key_choice = 0
                
    # Иногда нейросеть дописывает к запросу знаки, непонятно зачем. Убираем
    if text[0] and text[0] == '?' or text[0] and text[0] == '!':
        text = text[3:]
    
    return text # + model_info


async def edit_text(text, key_word):
    text_lower = text.lower()
    key_word_lower = key_word.lower()
    index = text_lower.find(key_word_lower)
    e_text = text[index + len(key_word):]
    
    return e_text