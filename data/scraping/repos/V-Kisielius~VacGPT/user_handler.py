import os
import openai
from functools import wraps
from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.filters import CommandObject
from dotenv import load_dotenv
load_dotenv()

ADMINS = list(map(int, os.environ['ADMINS'].split(',')))
SECRET_WORD = os.environ['SECRET_WORD']

router = Router()
openai.api_key = os.environ['OPENAI_TOKEN']

messages = {admin: [{'role': 'user', 'content': 'Привет! Называй меня Cэр!'}] for admin in ADMINS}

###################### HELPER ######################

def is_admin(func):
    @wraps(func)
    async def decorator(message: Message, *args, **kwargs):
        if message.from_user.id not in ADMINS:
            await message.answer('У Вас нет прав использовать эту команду!\nPermission denied!')
            return
        return await func(message, *args, **kwargs)
    return decorator

def is_user(func):
    @wraps(func)
    async def decorator(message: Message, *args, **kwargs):
        if message.from_user.id not in messages and message.from_user.id not in ADMINS:
            await message.answer('Вы не авторизованы!\nAuthentication error!')
            return
        return await func(message, *args, **kwargs)
    return decorator

###################### ADMIN ######################

@router.message(Command("password"))
@is_admin
async def show_SECRET_WORD(message: Message, command: CommandObject):
    new_pass = command.args
    if new_pass:
        global SECRET_WORD
        SECRET_WORD = new_pass
        await message.answer(f'New password is {SECRET_WORD}!')
        return
    await message.answer(f'Password: {SECRET_WORD}')
    return

@router.message(Command("users"))
@is_admin
async def show_users(message: Message):
    await message.answer('Users:')
    for user_id in messages:
        await message.answer(f'{user_id}: {messages[user_id][0]["content"]}')
    return

@router.message(Command("history"))
@is_admin
async def show_history(message: Message, command: CommandObject):
    user_id = command.args
    if user_id:
        try:
            user_id = int(user_id)
        except:
            await message.answer(f'User id must be a number!')
            return
        if user_id not in messages:
            await message.answer(f'User not found!')
            return
        await message.answer(f'History of {user_id}:')
        for message_ in messages[user_id]:
            await message.answer(message_['content'])
        return
    await message.answer(f'Usage: /history user_id: int')

###################### LOGIN ######################

@router.message(Command("login"))
async def login(message: Message, command: CommandObject):
    user_id = message.from_user.id
    user_name = message.from_user.full_name
    if user_id in ADMINS:
        await message.answer(f'Welcome, {user_name}! You are the creator, no need to log in -_-')
        if command.args != SECRET_WORD:
            await message.answer('But the password is wrong!')
        return
    if user_id in messages:
        await message.answer(f'Вы уже авторизованы!\nYou are already logged in!')
        return
    if command.args == SECRET_WORD:
        messages[user_id] = [{'role': 'user', 'content': f'Меня зовут {user_name}.'}]
        await message.answer(f'Добро пожаловать, {user_name}!\nWelcome, {user_name}!')
        await bot
        return
    else:
        await message.answer('Неверный пароль!\nWrong password!')
        return
        
###################### USER ######################
        
@router.message(Command("restart"))
@is_user
async def restart(message: Message):
    user_id = message.from_user.id
    user_name = message.from_user.full_name
    messages[user_id] = [{'role': 'user', 'content': f'Привет, меня зовут {user_name}!'}]
    await message.answer(f'Начнем сначала!\nLet\'s start over!')
    return

###################### TALK ######################

def update(history, role, content):
    history.append({'role': role, 'content': content})
    return history

@router.message()
@is_user
async def talk(message: Message):
    user_id = message.from_user.id
    messages[user_id] = update(messages[user_id], 'user', message.text)
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages[user_id])
    reply = response["choices"][0]["message"]["content"]
    await message.answer(reply)