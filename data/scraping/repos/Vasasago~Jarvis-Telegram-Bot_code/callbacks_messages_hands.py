import math
import os
import re
import subprocess
import sys
import webbrowser
from threading import Thread
from tkinter import messagebox

import keyboard
import openai
import psutil
import pyautogui
import requests
import speech_recognition as sr
from PIL import ImageGrab
from aiogram import types, Dispatcher
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import create_bot
import logger
import markups
import tts

max_tokens = create_bot.max_tokens
text_to_gpt = create_bot.text_to_gpt
output_file = create_bot.output_file
current_path = create_bot.current_path
page_number = create_bot.page_number
pages = create_bot.pages
drives_in = create_bot.drives_in
user_id = create_bot.user_id
names_drives = create_bot.names_drives
bot_version = create_bot.bot_version
gpt_model = create_bot.gpt_model
folders_names = create_bot.folders_names
root_folder = create_bot.root_folder
text_to_speech = create_bot.text_to_speech

name_folder = ''

dialog = []

link = ''

file_name = None

bot, dp = create_bot.create()

recognizer = sr.Recognizer()


# получаем экземпляры бота и диспетчера
def copy_bot():
    global bot, dp
    bot, dp = create_bot.create()


# получаем список дисков и кнопки
async def explore_disks():
    global user_id
    user_id = create_bot.user_id
    # Получаем список дисков, записываем в drives_in и создаём инлайн - кнопки
    drives = psutil.disk_partitions()
    drives_in.clear()

    # Проверяем диски на заполненность
    for drive in drives:
        try:
            drive_usage = psutil.disk_usage(drive.mountpoint)

            # Если объем диска больше 0, добавляем инлайн кнопку в массив
            if drive_usage.total > 0:
                drives_in.append(InlineKeyboardButton(drive.device, callback_data=drive.device))

        except Exception as e:
            logger.py_logger.error(f"{e}\n\n")

    # Создаем маркап с дисками
    drives_markup = InlineKeyboardMarkup(row_width=2).add(*drives_in, markups.back_to_pc_markup_btn)


    create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                      message_id=create_bot.edit_msg.message_id,
                                                      text=f'📂 Проводник\n💿 Выберите диск:',
                                                      reply_markup=drives_markup)

    names_drives.clear()

    # Записываем имена дисков в массив
    for name in drives_in:
        names_drives.append(name['callback_data'])


# вывод о боте
async def description():
    global user_id
    user_id = create_bot.user_id
    await bot.send_message(chat_id=user_id,
                           text=f"*Jarvis-Bot V{bot_version}*\n\n{create_bot.description}",
                           reply_markup=markups.service_markup, parse_mode="Markdown")


# проверка текста на содержание ссылки
def is_url(text):
    try:
        pattern = re.compile(r'https?://\S+')
        match = pattern.match(text)
        if bool(match):
            url = text
            req_response = requests.get(url)
            return True, req_response.status_code
        else:
            return (False,)
    except Exception as e:
        logger.py_logger.error(f"{e}\n\n")
        return (False,)


# проверка id пользователя
async def check_user_id(id_from_user):
    if str(id_from_user) != str(user_id):
        await bot.send_message(chat_id=id_from_user, text="❗ У вас нет доступа к этому боту!")
        return False
    else:
        return True


# Проводник: переходим по пути и генерируем Inline кнопки с названиями папок и файлов
async def explorer_func(number, page=1, items_per_page=20, query_id=''):
    # Объявляем путь, имя папки, номер страницы, общее количество страниц
    global current_path, name_folder, page_number, pages

    page_number = page  # Номер страницы


    # Формируем путь
    if number == '':  # Если имя папки не задано, берем диск из массива
        for name in names_drives:
            if current_path.replace('\\', '') in name.replace('\\', ''):
                current_path = current_path.replace('\\', '') + '\\'
                break
            else:
                current_path = current_path
                break

    elif current_path in names_drives:  # Если директория корневая (начало одного из дисков) прибавляем к диску папку
        name = folders_names.get(number)  # Получаем имя файла или папки по ее ключу
        current_path += f'{name}'
    else:
        name = folders_names.get(number)
        current_path += f'\\{name}'

    try:
        direct = os.listdir(current_path)  # Получаем список папок по пути

        folders = [] # Список папок

        for folder in direct:
            # Если папка не системная, добавляем ее в список
            if folder[0] != '.' and folder[0] != '$':
                folders.append(folder)

        create_bot.console += f'directory: {current_path} page: {page_number}\n' # Выводим путь и номер страницы в консоль


        # Рассчитываем начальный и конечный индексы для текущей страницы
        start_index = (page - 1) * items_per_page
        end_index = start_index + items_per_page

        pages = math.ceil((len(folders) / items_per_page)) # Рассчитываем количество страниц


        inline_folders = []  # Пустой массив для инлайн кнопок с названиями папок и коллбэками в виде их ключей
        folders_names.clear()

        i = 0

        # Создаем список с Inline-кнопками только для элементов на текущей странице
        for folder in folders[start_index:end_index]:
            #  Если хотим получить список программ (меню -> программы)
            if query_id == '0' or 'lnk' in folder or ' - Ярлык.lnk' in folder:
                name_folder = folder.replace('.lnk', '').replace('.exe', '')

            #  Меняем название папки на users
            elif folder.lower() == 'пользователи' or folder.lower() == '%1$d пользователей':
                name_folder = 'Users'

            # Присваиваем имя папки
            else:
                name_folder = folder

            # Если имя папки длиннее 20 символов, укорачиваем его
            if len(name_folder) > 20:
                name_folder = name_folder[:10] + '...' + name_folder[-10:]

            # Добавляем в массив кнопку с папкой
            inline_folders.append(InlineKeyboardButton(f'{name_folder}', callback_data=str(i)))
            # Добавляем папку в словарь по ее ключу
            folders_names[str(i)] = folder
            i += 1

        # Создаем маркап с кнопками папок
        folders_markup = InlineKeyboardMarkup(row_width=2).add(*inline_folders)

        # Создаем кнопки для переключения между страницами
        previous_button = InlineKeyboardButton('◀ Предыдущая страница', callback_data='previous_page')
        next_button = InlineKeyboardButton('Следующая страница ▶', callback_data='next_page')

        # Добавляем кнопки в маркап
        if page == 1 and pages > 1:
            folders_markup.row(next_button)
        elif end_index >= len(folders) and pages > 1:
            folders_markup.row(previous_button)
        elif pages <= 1:
            pass
        else:
            folders_markup.row(previous_button, next_button)


        # Если находимся не в папке программ, добавляем кнопки возврата
        if query_id != '0':
            # Если путь это диск из массива
            if current_path in names_drives:
                go_back_to_drives = InlineKeyboardButton('◀ К дискам', callback_data='back_to_drives')
                folders_markup.row(go_back_to_drives)
            else:
                go_back_to_drives = InlineKeyboardButton('◀ К дискам', callback_data='back_to_drives')
                go_back_explorer = InlineKeyboardButton('◀ Назад', callback_data='back_explorer')
                folders_markup.row(go_back_explorer, go_back_to_drives)

        if query_id != '0':
            await bot.answer_callback_query(callback_query_id=query_id)

        return current_path, folders_markup  # Возвращаем путь и Маркапы

    except PermissionError as e:
        create_bot.console += f'\nОшибка explorer_func: {e}\n\n'
        logger.py_logger.error(f"{e}\n\n")

        await bot.answer_callback_query(callback_query_id=query_id, text="❗ Отказано в доступе.", show_alert=True)
        current_path = os.path.dirname(current_path)

    except FileNotFoundError as e:
        create_bot.console += f'\nОшибка explorer_func: {e}\n\n'
        logger.py_logger.error(f"{e}\n\n")

        await bot.answer_callback_query(callback_query_id=query_id, text="❗ Файл не найден.", show_alert=True)
        await explore_disks()

    except Exception as e:
        create_bot.console += f'\nОшибка explorer_func: {e}\n\n'
        logger.py_logger.error(f"{e}\n\n")

        await bot.answer_callback_query(callback_query_id=query_id, text="❗ Произошла ошибка.", show_alert=True)
        await explore_disks()


# хендлер текстовых кнопок
async def text_markups(message: types.Message):
    global user_id

    user_id = create_bot.user_id

    if await check_user_id(message.from_user.id):

        if message.text == '🤖 Команды Jarvis':
            # Выводим папки с командами
            await bot.send_message(chat_id=user_id, text='📂 Выберите папку:',
                                   reply_markup=markups.open_commands())

        elif message.text == '🖥 Компьютер':
            # Вывод меню компьютера
            create_bot.edit_msg = await bot.send_message(chat_id=user_id, text='👉 Выберите действие:',
                                                         reply_markup=markups.pc_markup)

        elif message.text == '🛠 Управление ботом':
            await description()


# хендлер gpt и ссылок
async def all_messages(message: types.Message):
    global user_id, max_tokens, text_to_gpt, dialog

    user_id = create_bot.user_id

    if await check_user_id(message.from_user.id):

        # Получаем ответ от сервера
        response_func = is_url(message.text)

        # Коды ошибок
        errors_codes = {
            201: 'Created',
            204: 'No Content (сервер не возвращает никаких данных)',
            301: 'Moved Permanently (ресурс был перемещен на другой адрес)',
            400: 'Bad Request (запрос имеет неверный формат)',
            401: 'Unauthorized (необходима авторизация)',
            403: 'Forbidden (недостаточно прав)',
            404: 'Not Found (ресурс не найден)',
            500: 'Internal Server Error (ошибка на сервере)'
        }

        # Если сообщение содержит ссылку, открываем ее в браузере
        if response_func[0]:
            # Если статус-код = ОК
            if response_func[1] == 200:
                create_bot.console += f'link: {message.text}\n'
                await message.answer("✅ Ссылка отправлена!", reply_markup=markups.main_inline)
                webbrowser.open(url=message.text)

            # Если возникла ошибка при запросе
            else:
                global link
                link = message.text
                create_bot.console += f'link: {message.text} error: {response_func[1]}\n'
                create_bot.edit_msg = await message.answer(f"❗ Не удалось выполнить запрос. Код ошибки:"
                                     f" {response_func[1]} - {errors_codes[response_func[1]]}\n"
                                     f"Вы хотите открыть ссылку?",
                                     reply_markup=markups.open_link_markup)

        # Если перемещаем мышь
        elif len(message.text.split()) == 2 and message.text.split()[0].lower() in ['вверх', 'вниз', 'влево', 'вправо']:
            try:
                # Разбираем направление и расстояние из сообщения пользователя
                direction, distance = message.text.split()
                distance = int(distance)

                # Выполняем перемещение мыши в указанном направлении и расстоянии
                if direction.lower() == 'вверх':
                    pyautogui.move(0, -distance)
                elif direction.lower() == 'вниз':
                    pyautogui.move(0, distance)
                elif direction.lower() == 'влево':
                    pyautogui.move(-distance, 0)
                elif direction.lower() == 'вправо':
                    pyautogui.move(distance, 0)
                else:
                    raise ValueError

                await bot.send_message(chat_id=user_id, text=f"Мышь перемещена {direction.lower()}"
                                                             f" на {distance} пикселей.")
            except (ValueError, IndexError):
                await bot.send_message(chat_id=user_id,
                                       text="Ошибка! Пожалуйста, введите направление (вверх, вниз, влево, вправо)"
                                            " и расстояние в правильном формате.")

        elif len(message.text.split()) == 1 and message.text.lower() in ['пкм', 'лкм']:
            if message.text.lower() == 'пкм':
                pyautogui.click(button='right')
            elif message.text.lower() == 'лкм':
                pyautogui.click(button='left')
            else:
                return


        # Запрос к GPT
        else:
            create_bot.edit_msg = await bot.send_message(chat_id=user_id, text="⏳ Ваш запрос отправлен.")

            dialog.append({"role": "user", "content": message.text}) # Добавляем запрос в историю диалога

            try:
                create_bot.console += f'ChatGPT: {message.text}.\n'

                # Генерируем ответ
                completion = openai.ChatCompletion.create(model=gpt_model, messages=dialog)

                # Отправляем ответ пользователю
                response = '🤖 Jarvis:\n' + completion.choices[0].message.content

                create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                  message_id=create_bot.edit_msg.message_id,
                                                                  text=response, reply_markup=markups.gpt_markup)

                # Добавляем ответ в историю диалога
                dialog.append({"role": "assistant", "content": response.replace('🤖 Jarvis:\n', '')})


            except openai.error.TryAgain as e:
                create_bot.console += f'\nОшибка gpt: {e}\n\n'
                logger.py_logger.error(f"{e}\n\n")

                await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                            text='🫡Не удалось выполнить запрос. Попробуйте снова.')

            # Обработка других исключений openai.error
            except Exception as e:
                create_bot.console += f'\nОшибка gpt: {e}\n\n'

                logger.py_logger.error(f"{e}\n\n")

                await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                            text='🫡Не удалось выполнить запрос. Подробнее читайте в Консоли.')


# Открытие ссылки игнорируя ошибку
async def open_link(callback_query: types.CallbackQuery):
    global link

    await bot.answer_callback_query(callback_query.id)
    webbrowser.open(url=link)
    await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                text="✅ Ссылка отправлена!")



# меню компьютера
async def computer_menu(callback_query: types.CallbackQuery):
    global user_id, current_path

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        def convert_bytes_to_gigabytes(bytes_value):
            gigabytes = bytes_value / (1024 ** 3)
            return gigabytes

        def get_disk_usage():
            try:
                disk_partitions = psutil.disk_partitions()
                disk_usages = {}
                for partition in disk_partitions:
                    usages = psutil.disk_usage(partition.mountpoint)
                    disk_usages[partition.device] = {
                        'total': usages.total,
                        'used': usages.used,
                        'free': usages.free,
                        'percent': usages.percent
                    }

                return disk_usages

            except Exception as e:
                logger.py_logger.error(f"{e}\n\n")

        def get_system_load():
            cpu_percents = psutil.cpu_percent(interval=1)
            memory_percents = psutil.virtual_memory().percent
            ram_used = psutil.virtual_memory().used
            ram_total = psutil.virtual_memory().total

            return cpu_percents, (memory_percents, ram_total, ram_used)

        if command == 'pc_control':
            await bot.answer_callback_query(callback_query.id)

            disk_usage = get_disk_usage()
            cpu_percent, memory = get_system_load()

            drives = ''

            for disk, usage in disk_usage.items():
                total_gb = convert_bytes_to_gigabytes(usage['total'])
                used_gb = convert_bytes_to_gigabytes(usage['used'])
                free_gb = convert_bytes_to_gigabytes(usage['free'])
                percent = usage['percent']

                drives += f"Диск: {disk}\n"
                drives += f"Общий объем: {total_gb:.2f} ГБ\n"
                drives += f"Использовано: {used_gb:.2f} ГБ\n"
                drives += f"Свободно: {free_gb:.2f} ГБ\n"
                drives += f"Заполненность: {percent}%\n\n"


            memory_total = convert_bytes_to_gigabytes(memory[1])
            memory_used = convert_bytes_to_gigabytes(memory[2])

            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text=f"💿 ROM:\n{drives}"
                                                                   f"📈 CPU загруженность: {cpu_percent}%\n\n"
                                                                   f"📈 RAM загруженность: {memory[0]}%\n"
                                                                   f"📈 RAM всего: {memory_total:.2f}ГБ\n"
                                                                   f"📈 RAM использовано: {memory_used:.2f}ГБ",
                                                              reply_markup=markups.back_to_pc_markup)

        if command == 'keyboard':
            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text='⌨ Клавиатура\nВыберите действие:',
                                                              reply_markup=markups.keyboard_inline)


        if command == 'mouse':
            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text='🐁 Мышь\nВыберите действие:',
                                                              reply_markup=markups.Mouse_markup)


        if command == 'explorer':
            await explore_disks()


        if command == 'programs':
            current_path = os.path.dirname(os.path.abspath(sys.argv[0])) + '\\lnk'
            result = await explorer_func(number='', query_id='0')

            if result is not None and pages >= 1:
                folder, buttons = result

                buttons.add(markups.back_to_pc_markup_btn)

                create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                  message_id=create_bot.edit_msg.message_id,
                                                                  text=f'🖥 Программы:',
                                                                  reply_markup=buttons)

            else:
                create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                  message_id=create_bot.edit_msg.message_id,
                                                                  text=f'🖥 В данной папке нет программ.'
                                                                       f' Добавьте их ярлыки или сами программы'
                                                                       f' в папку lnk по этому пути:\n'
                                                                       f'{current_path}',
                                                                  reply_markup=markups.open_lnk_markup)

        if command == 'open_lnk':
            await bot.answer_callback_query(callback_query.id)
            lnk_path = os.path.dirname(os.path.abspath(sys.argv[0])) + '\\lnk'
            os.system(f"explorer.exe {lnk_path}")


        if command == 'commands_windows':
            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text='👉 Выберите действие:',
                                                              reply_markup=markups.commands_windows(0))

        if command == 'tasks':
            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text='💽 Диспетчер задач\n👉 Выберите приложение, которое вы хотите закрыть:',
                                                              reply_markup=markups.tasks()[1])


        if command == 'back_pc':
            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text='👉 Выберите действие:',
                                                              reply_markup=markups.pc_markup)


async def terminate_progs(callback_query: types.CallbackQuery):
    global user_id

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):
        command = callback_query.data

        for proc in psutil.process_iter(['name']):
            try:
                proc.name()
                proc_name = proc.name()
                if proc_name == command:
                    proc.kill()  # Завершение процесса
                    create_bot.console += f'Kill process: {command}\n'

                    await bot.answer_callback_query(callback_query_id=callback_query.id,
                                                    text="✅ Приложение закрыто!", show_alert=False)

                    create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                      message_id=create_bot.edit_msg.message_id,
                                                                      text='💽 Диспетчер задач\n👉 Выберите приложение, которое вы хотите закрыть:',
                                                                      reply_markup=markups.tasks()[1])

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                await bot.answer_callback_query(callback_query_id=callback_query.id,
                                                text="❗️ Не удалось завершить процесс!", show_alert=False)


async def commands_windows_handler(callback_query: types.CallbackQuery):
    global user_id

    user_id = create_bot.user_id

    page = 0

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        async def send_message():
            create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                              message_id=create_bot.edit_msg.message_id,
                                                              text='👉 Выберите действие:',
                                                              reply_markup=markups.commands_windows(page))

        if command == 'next':
            page = 1
            await send_message()

        elif command == 'back':
            page = 0
            await send_message()

        else:
            create_bot.console += f'subprocess: Windows_Commands/{command}\n'

            await bot.answer_callback_query(callback_query_id=callback_query.id,
                                            text="✅ Выполнено!", show_alert=False)

            if command == 'screenshot.exe':
                create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                  message_id=create_bot.edit_msg.message_id,
                                                                  text='⏳ Идёт загрузка скриншота.')

                path = 'screenshot.png'
                screenshot = ImageGrab.grab()
                screenshot.save(path, 'PNG')

                await bot.send_document(chat_id=user_id, document=open(path, 'rb'))

                os.remove('screenshot.png')

                await bot.delete_message(chat_id=user_id, message_id=create_bot.edit_msg.message_id)
                create_bot.edit_msg = await bot.send_message(chat_id=user_id, text='👉 Выберите действие:',
                                                                  reply_markup=markups.commands_windows(page))

            subprocess.run([f'Windows_Commands/{command}'])



# озвучка
async def silero_tts(callback_query: types.CallbackQuery):
    global user_id, current_path

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        def what_speaker(cmd):
            name_speaker = create_bot.speaker[int(cmd.split('-')[1])]

            if name_speaker == 'aidar':
                return 'Айдар'

            elif name_speaker == 'baya':
                return 'Байя'

            elif name_speaker == 'kseniya':
                return 'Ксения 1'

            elif name_speaker == 'xenia':
                return 'Ксения 2'

            else:
                return 'Евгений'

        def check_model():
            if os.path.isfile('model.pt'):
                size = os.path.getsize('model.pt')
            else:
                size = 0

            if size < 61896251:
                if messagebox.askokcancel("Модель не найдена", "Модель Silero TTS не найдена. Вы хотите ее установить?"):
                    tts.is_run = True
                    Thread(target=tts.start_tts(), name='tts')
                    return True

                else:
                    return False
            else:
                return True

        if command.split('-')[0] == 'voice':
            await bot.answer_callback_query(callback_query_id=callback_query.id)
            await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                        text=f"✅ Текст отправлен!\n🗣 Голос: {what_speaker(command)}.")

            if check_model():
                try:
                    await bot.send_voice(chat_id=user_id,
                                         voice=tts.va_speak(what=create_bot.text_to_speech,
                                                            voice=True,
                                                            speaker=create_bot.speaker[int(command.split('-')[1])]))

                    os.remove('audio.mp3')

                except Exception as e:
                    logger.py_logger.error(f"{e}\n\n")

        if command.split('-')[0] == 'audio':
            await bot.answer_callback_query(callback_query_id=callback_query.id)
            await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                        text=f"✅ Текст отправлен!\n🗣 Голос: {what_speaker(command)}.")

            if check_model():
                tts.va_speak(what=create_bot.text_to_speech, voice=False,
                             speaker=create_bot.speaker[int(command.split('-')[1])])


# закончить диалог с gpt
async def gpt_close_dialog(callback_query: types.CallbackQuery):
    global text_to_gpt, user_id, current_path, dialog

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        try:
            await bot.answer_callback_query(callback_query_id=callback_query.id)
            await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                        text=create_bot.edit_msg.text, reply_markup=None)
            dialog.clear()
            await bot.send_message(chat_id=user_id, text='✅ Вы закончили диалог.')
        except Exception as e:
            logger.py_logger.error(f"{e}\n\n")


# перевод из гс в текст
async def recognize_voice(callback_query: types.CallbackQuery):
    global user_id, output_file
    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        await bot.answer_callback_query(callback_query_id=callback_query.id)

        lang = callback_query.data

        create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                                          text=f'⏳ Идёт распознавание.')

        if lang == 'RU-ru':
            lang_sticker = '🇷🇺'
        elif lang == 'UK-uk':
            lang_sticker = '🇺🇦'
        else:
            lang_sticker = '🇺🇸'

        try:
            with sr.AudioFile(output_file) as audio:
                audio_data = recognizer.record(audio)
                text = recognizer.recognize_google(audio_data, language=lang)
                create_bot.console += f'speech to text: {text}\n'

                await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                            text=f'📝{lang_sticker}Распознанный текст:\n{text}.')

        except sr.exceptions.UnknownValueError:
            create_bot.console += '\nОшибка при распознавании голосового сообщения.\n\n'

            await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                        text=f'🫡При распознавании возникла ошибка.')

        except Exception as e:
            create_bot.console += f'\nОшибка при распознавании голосового сообщения: {e}\n\n'

        os.remove(output_file)


# функции бота
async def bot_settings(callback_query: types.CallbackQuery):
    global user_id, current_path

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        if command == 'bot_path':
            current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
            result = await explorer_func(number='', query_id=callback_query.id)

            if result is not None:
                folder, buttons = result
                await bot.edit_message_text(chat_id=user_id,
                                            message_id=callback_query.message.message_id,
                                            text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                 f' из {pages}\n➡ Текущий путь:\n{folder}', reply_markup=buttons)

        if command == 'log':
            await bot.answer_callback_query(callback_query.id)
            await bot.delete_message(chat_id=user_id, message_id=callback_query.message.message_id)
            await bot.send_message(chat_id=user_id, text=f'⏳ Идет загрузка лога.')
            create_bot.console += f'download log-file\n'
            with open('logs_from_bot.log', 'rb') as log_file:
                await bot.send_document(chat_id=user_id, document=log_file)

            await description()

        if command == 'start_voice_jarvis':
            await bot.answer_callback_query(callback_query.id)
            create_bot.edit_msg = await bot.send_message(chat_id=user_id, text='🖥 Запускаю голосового Jarvis...')

            try:
                subprocess.Popen('start-voice-jarvis.exe')
                await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                            text='✅ Голосовой Jarvis запущен.')

            except Exception as e:
                logger.py_logger.error(f"{e}\n\n")
                await bot.edit_message_text(chat_id=user_id, message_id=create_bot.edit_msg.message_id,
                                            text='❗ Не удалось запустить голосового Jarvis. Убедитесь,'
                                                 ' что в папке бота присутствует файл start-voice-jarvis.exe.')

        if command == 'off':
            await bot.answer_callback_query(callback_query.id)
            await bot.send_message(chat_id=user_id, text='📴 Выключение...')
            subprocess.Popen('off.exe')

        if command == 'reboot':
            await bot.answer_callback_query(callback_query.id)
            await bot.send_message(chat_id=user_id, text='♻ Перезагрузка...')
            subprocess.Popen('reboot.exe')


# навигация по проводнику
async def explorer_navigation(callback_query: types.CallbackQuery):
    global user_id, page_number, current_path

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        if command == 'next_page':
            page_number = page_number + 1

            result = await explorer_func(number='', page=page_number, query_id=callback_query.id)

            if result is not None:
                folder, buttons = result
                await bot.edit_message_text(chat_id=user_id,
                                            message_id=callback_query.message.message_id,
                                            text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                 f' из {pages}\n➡ Текущий путь:\n{folder}', reply_markup=buttons)

        if command == 'previous_page':
            page_number = page_number - 1

            result = await explorer_func(number='', page=page_number, query_id=callback_query.id)

            if result is not None:
                folder, buttons = result
                await bot.edit_message_text(chat_id=user_id,
                                            message_id=callback_query.message.message_id,
                                            text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                 f' из {pages}\n➡ Текущий путь:\n{folder}', reply_markup=buttons)



        if command == 'back_to_drives' or command == 'back_explorer':
            try:
                if command == 'back_explorer':
                    current_path = os.path.dirname(current_path)
                    result = await explorer_func(number='', query_id=callback_query.id)

                    if result is not None:
                        folder, buttons = result
                        await bot.edit_message_text(chat_id=user_id,
                                                    message_id=callback_query.message.message_id,
                                                    text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                         f' из {pages}\n➡ Текущий путь:\n{folder}',
                                                    reply_markup=buttons)

                    else:
                        pass

                else:
                    await explore_disks()

            except Exception as e:
                create_bot.console += f'\nОшибка при попытке вернуться на директорию выше: {e}\n\n'
                logger.py_logger.error(f"{e}\n\n")
                await explore_disks()


# действия с файлами
async def actions_with_files(callback_query: types.CallbackQuery):
    global user_id, page_number, current_path

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        if command == 'run':
            create_bot.console += f'subprocess: {current_path}\n'

            subprocess.run(['start', '', current_path], shell=True)

            await bot.answer_callback_query(callback_query_id=callback_query.id,
                                            text="✅ Выполнено!", show_alert=False)

        if command == 'download':
            current_path = os.path.dirname(current_path)
            result = await explorer_func(number='', query_id=callback_query.id)
            if result is not None:
                folder, buttons = result
                try:
                    create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                      message_id=callback_query.message.message_id,
                                                                      text='⏳ Идёт загрузка файла.')

                    file_path_name = ''

                    for name in names_drives:
                        if current_path in name:
                            file_path_name = current_path + f'{file_name}'
                            break
                        else:
                            file_path_name = current_path + f'\\{file_name}'
                            break

                    with open(file_path_name, 'rb') as file:
                        create_bot.console += f'upload file: {file_name}\n'
                        await bot.send_document(chat_id=user_id, document=file)
                        create_bot.edit_msg = await bot.send_message(chat_id=user_id,
                                                                     text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                                          f' из {pages}\n➡ Текущий путь:\n{folder}',
                                                                     reply_markup=buttons)

                except Exception as e:
                    await bot.edit_message_text(chat_id=user_id,
                                                message_id=create_bot.edit_msg.message_id,
                                                text='🫡При загрузке файла возникла ошибка.'
                                                     ' Подробнее читайте в Консоли.')
                    create_bot.edit_msg = await bot.send_message(chat_id=user_id,
                                                                 text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                                      f' из {pages}\n➡ Текущий путь:\n{folder}',
                                                                 reply_markup=buttons)

                    create_bot.console += f'\nОшибка handle_callback (попытка отправить файл): {e}\n\n'

            else:
                pass

        if command == 'delete':
            create_bot.console += f'delete: {current_path}\n'

            os.remove(current_path)

            await bot.answer_callback_query(callback_query_id=callback_query.id,
                                            text="✅ Файл удален!", show_alert=False)


# возвращает к папкам с командами
async def back_to_commands_folder(callback_query: types.CallbackQuery):
    global user_id

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data

        if command == 'commands':
            await bot.answer_callback_query(callback_query.id)
            await bot.edit_message_text(chat_id=user_id, message_id=callback_query.message.message_id,
                                        text='📂 Выберите папку:', reply_markup=markups.open_commands())


# основные функции проводника
async def main_explorer(callback_query: types.CallbackQuery):
    global current_path, page_number, pages, file_name, user_id

    user_id = create_bot.user_id

    if await check_user_id(callback_query.from_user.id):

        command = callback_query.data
        names_dict = {}

        def read_text_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                return lines

            except Exception as e:
                logger.py_logger.error(f"{e}\n\n")

        def scan_folders(root_folder):
            exe_files = {}
            for foldername, subfolders, filenames in os.walk(root_folder):
                if foldername.endswith("ahk"):
                    for filename in filenames:
                        if filename.endswith(".exe"):
                            exe_path = os.path.join(foldername, filename)
                            exe_files[filename] = exe_path
            return exe_files

        folders = os.listdir(root_folder)

        if folders:
            for foldername, subfolders, filenames in os.walk(root_folder):
                for filename in filenames:
                    if filename == "names.txt":
                        file_path = os.path.join(foldername, filename)
                        lines = read_text_file(file_path)
                        if lines:
                            for line in lines:
                                line = line.strip()
                                names_dict[line.split(':')[1]] = line.split(':')[0]

        exe_files = scan_folders(root_folder)

        if command.startswith('folder:'):
            await bot.answer_callback_query(callback_query.id)
            folder_name = command.split(':')[1]
            subfolder_path = os.path.join(root_folder, folder_name, 'ahk')
            exe_files = scan_folders(subfolder_path)

            if exe_files:
                global files
                files = []
                files.clear()
                for filename in exe_files.keys():
                    for key, val in names_dict.items():
                        if str(filename.split('.')[0]) == key:
                            files.append(InlineKeyboardButton(val, callback_data=filename))
                        elif str(filename.split('.')[0]) not in names_dict.keys():
                            if InlineKeyboardButton(filename, callback_data=filename) not in files:
                                files.append(InlineKeyboardButton(filename, callback_data=filename))

                inline_files = InlineKeyboardMarkup(row_width=2).add(*files, InlineKeyboardButton('🔙 Вернуться назад',
                                                                                                  callback_data=
                                                                                                  'commands'))
                await bot.edit_message_text(chat_id=user_id,
                                            message_id=callback_query.message.message_id,
                                            text=f'📂 Текущая папка: {folder_name}.\nВыберите действие:',
                                            reply_markup=inline_files)
            else:
                await bot.edit_message_text(chat_id=user_id,
                                            message_id=callback_query.message.message_id,
                                            text='✖ В данной папке нет файлов.', reply_markup=markups.open_commands())

        if command in names_drives:
            current_path = command
            try:
                result = await explorer_func(number='', query_id=callback_query.id)

                if result is not None:
                    folder, buttons = result
                    if pages >= 1:
                        await bot.edit_message_text(chat_id=user_id,
                                                    message_id=callback_query.message.message_id,
                                                    text=f'📂 Проводник\n📃 Страница:\n{page_number}'
                                                         f' из {pages}\n➡ Текущий путь: {folder}', reply_markup=buttons)

                    else:
                        go_back_explorer = InlineKeyboardButton('◀ Назад', callback_data='back_explorer')
                        folders_markup = InlineKeyboardMarkup(row_width=1).add(go_back_explorer)
                        await bot.edit_message_text(chat_id=user_id,
                                                    message_id=callback_query.message.message_id,
                                                    text=f'📂 Проводник\n➡ Текущий путь:\n{folder}\n'
                                                         f'✖ В данной папке нет файлов.', reply_markup=folders_markup)
                else:
                    pass

            except Exception as e:
                logger.py_logger.error(f"{e}\n\n")


        if command in folders_names.keys():
            if os.path.isdir(current_path + f'\\{folders_names.get(command)}'):
                try:
                    create_bot.console += f'folder: {folders_names.get(command)}\n'

                    result = await explorer_func(number=command, query_id=callback_query.id)

                    if result is not None:
                        folder, buttons = result
                        if pages >= 1:
                            await bot.edit_message_text(chat_id=user_id,
                                                        message_id=callback_query.message.message_id,
                                                        text=f'📂 Проводник\n📃 Страница: {page_number}'
                                                             f' из {pages}\n➡ Текущий путь:\n{folder}',
                                                        reply_markup=buttons)

                        else:
                            go_back_explorer = InlineKeyboardButton('◀ Назад', callback_data='back_explorer')
                            folders_markup = InlineKeyboardMarkup(row_width=1).add(go_back_explorer)
                            await bot.edit_message_text(chat_id=user_id,
                                                        message_id=callback_query.message.message_id,
                                                        text=f'📂 Проводник\n➡ Текущий путь:\n{folder}\n'
                                                             f'✖ В данной папке нет файлов.',
                                                        reply_markup=folders_markup)

                except Exception as e:
                    if current_path not in names_drives:
                        index = current_path.rfind('\\')
                        if index != -1:
                            current_path = current_path[:index]
                            result = await explorer_func(number='', query_id=callback_query.id)

                            if result is not None:
                                folder, buttons = result
                                await bot.edit_message_text(chat_id=user_id,
                                                            message_id=callback_query.message.message_id,
                                                            text=f'📂 Проводник\n🫡Не удалось открыть папку.\n📃 Страница:'
                                                                 f' {page_number} из {pages}\n➡ Текущий путь:\n{folder}',
                                                            reply_markup=buttons)

                                create_bot.console += f'\nОшибка при попытке открыть папку: {e}\n\n'
                                logger.py_logger.error(f"{e}\n\n")

                            else:
                                pass

            else:
                file_name = folders_names.get(command)
                if current_path == os.path.dirname(os.path.abspath(sys.argv[0])) + '\\lnk':

                    create_bot.console += f'subprocess: {current_path}\\{file_name}\n'

                    subprocess.run(['start', '', current_path + f'\\{file_name}'], shell=True)
                    await bot.answer_callback_query(callback_query_id=callback_query.id,
                                                    text="✅ Выполнено!", show_alert=False)

                else:
                    current_path = current_path + '\\' + file_name
                    if os.path.exists(current_path):
                        create_bot.edit_msg = await bot.edit_message_text(chat_id=user_id,
                                                                          message_id=callback_query.message.message_id,
                                                                          text=f'➡ Текущий путь:\n{current_path}'
                                                                               + '\n📂 Выберите действие:',
                                                                          reply_markup=markups.script_file_markup)
                    else:
                        await bot.answer_callback_query(callback_query_id=callback_query.id,
                                                        text="❗ Устройство не найдено.", show_alert=True)
                        await explore_disks()


        async def keyboard_press(btn):
            if command == btn.callback_data:
                await bot.answer_callback_query(callback_query.id)
                create_bot.console += f'keyboard press: {command}\n'
                keyboard.press_and_release(command)

        for btn1, btn2 in zip(markups.keys, markups.f):
            await keyboard_press(btn1)
            await keyboard_press(btn2)

        for mouse_btn in markups.mouse_btns:
            if command == mouse_btn.callback_data:
                await bot.answer_callback_query(callback_query.id)
                # Разбираем направление и расстояние из сообщения пользователя
                direction, distance = command.split('_')
                distance = int(distance)

                # Выполняем перемещение мыши в указанном направлении и расстоянии
                if direction == 'left' and distance == 0:
                    pyautogui.click(button='left')
                elif direction == 'right' and distance == 0:
                    pyautogui.click(button='right')
                elif direction == 'up':
                    pyautogui.move(0, -distance)
                elif direction == 'down':
                    pyautogui.move(0, distance)
                elif direction == 'left':
                    pyautogui.move(-distance, 0)
                elif direction == 'right':
                    pyautogui.move(distance, 0)
                else:
                    return

        for key, val in exe_files.items():
            if command == key:
                create_bot.console += 'subprocess: {}\\{}\n'.format(val.split("\\")[-3], command)
                subprocess.Popen(val)

# регистрируем хенд леры
def callbacks_messages_handlers(dispatcher: Dispatcher):
    try:
        # messages
        dispatcher.register_message_handler(text_markups, lambda message: message.text in
                                                                       ['🤖 Команды Jarvis', '🖥 Компьютер',
                                                                        '🛠 Управление ботом'])

        dispatcher.register_message_handler(all_messages)


        # callbacks
        dispatcher.register_callback_query_handler(open_link, lambda c: c.data == 'open_link')

        dispatcher.register_callback_query_handler(computer_menu, lambda c: c.data in ['pc_control', 'keyboard',
                                                                                       'mouse', 'commands_windows', 'explorer', 'programs',
                                                                                       'open_lnk', 'back_pc', 'tasks'])

        dispatcher.register_callback_query_handler(commands_windows_handler, lambda c: c.data in [btn1.callback_data for btn1 in markups.commands_windows_btns1] +
                                                             [btn2.callback_data for btn2 in markups.commands_windows_btns2] + [markups.go_next.callback_data, markups.go_back.callback_data])

        dispatcher.register_callback_query_handler(terminate_progs, lambda c: c.data in [btn.callback_data for btn in markups.tasks()[0]])

        dispatcher.register_callback_query_handler(silero_tts, lambda c: c.data.startswith(('audio', 'voice')))

        dispatcher.register_callback_query_handler(gpt_close_dialog, lambda c: c.data=='close_dialog')

        dispatcher.register_callback_query_handler(recognize_voice, lambda c: c.data in ['RU-ru', 'UK-uk', 'EN-en'])

        dispatcher.register_callback_query_handler(bot_settings, lambda c: c.data in ['bot_path', 'log', 'start_voice_jarvis',
                                                                              'off', 'reboot'])

        dispatcher.register_callback_query_handler(explorer_navigation, lambda c: c.data in
                                                                                ['next_page', 'previous_page',
                                                                                 'back_to_drives', 'back_explorer'])

        dispatcher.register_callback_query_handler(actions_with_files, lambda c: c.data in ['run', 'download', 'delete'])

        dispatcher.register_callback_query_handler(back_to_commands_folder, lambda c: c.data=='commands')

        dispatcher.register_callback_query_handler(main_explorer)

    except Exception as e:
        logger.py_logger.error(f"{e}\n\n")
