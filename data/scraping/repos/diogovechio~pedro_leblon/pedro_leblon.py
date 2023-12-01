import asyncio
import logging
import os
import random
import sys
from asyncio import AbstractEventLoop, Semaphore

from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import aiohttp
import json

import face_recognition

import schedule

import typing as T

from aiohttp import ClientSession

from constants.constants import SECRETS_FILE
from data_classes.bot_config import BotConfig
from data_classes.commemorations import Commemorations
from data_classes.received_message import MessagesResults, TelegramMessage, MessageReceived
from data_structures.max_size_list import MaxSizeList
from messages_reactions import messages_coordinator
from utils.logging_utils import telegram_logging, elapsed_time, async_elapsed_time
from utils.openai_utils import OpenAiCompletion
from utils.text_utils import create_username
from utils.text_utils import send_message_last_try
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)

session_timeout = aiohttp.ClientTimeout(
    total=None,
    sock_connect=120,
    sock_read=120
)


class FakePedro:
    def __init__(
            self,
            bot_config_file: str,
            commemorations_file: str,
            user_mood_file: str,
            user_opinions_file: str,
            secrets_file: str,
            polling_rate: int = 1,
            debug_mode=False
    ):
        self.allowed_list = []
        self.debug_mode = debug_mode

        self.config: T.Optional[BotConfig] = None
        self.config_file = bot_config_file
        self.commemorations_file = commemorations_file
        self.user_mood_file = user_mood_file
        self.user_opinions_file = user_opinions_file
        self.commemorations: T.Optional[Commemorations] = None
        self.secrets_file = secrets_file

        self.semaphore = Semaphore(1)

        self.last_id = 0
        self.polling_rate = polling_rate
        self.messages: T.List[T.Any] = []
        self.interacted_updates = MaxSizeList(400)
        self.interacted_messages_with_chat_id = MaxSizeList(400)

        self.messages_in_memory = defaultdict(lambda: MaxSizeList(130))  # legacy
        self.chats_in_memory = defaultdict(list)
        self.chat_in_memory_max_load_days = 180

        self.mood_per_user = defaultdict(lambda: 0.0)
        self.user_opinions = defaultdict(list)

        self.datetime_now = datetime.now() - timedelta(hours=3)

        self.schedule = schedule

        self.api_route = ""
        self.session: T.Optional[ClientSession] = None

        self.face_images_path = 'faces'
        self.alpha_faces_path = 'faces_alpha'
        self.faces_names = []
        self.faces_files = []
        self.alpha_faces_files = []
        self.face_embeddings = []

        self.dall_e_uses_today = []

        self.asked_for_photo = 0

        self.mocked_hour = 0
        self.random_talk = 0
        self.kardashian_gif = 0
        self.mocked_today = False
        self.sent_news = 0

        self.messages_tasks = defaultdict(lambda: MaxSizeList(15))

        self.roleta_hour = 14
        self.last_roleta_day = 0

        self.openai: T.Optional[OpenAiCompletion] = None

        self.loop: T.Optional[AbstractEventLoop] = None

    async def run(self) -> None:
        try:
            from scheduling import scheduler

            Path('tmp').mkdir(exist_ok=True)
            Path('chat_logs').mkdir(exist_ok=True)
            Path('face_lake').mkdir(exist_ok=True)
            Path('image_tasks').mkdir(exist_ok=True)
            Path('image_tasks_done').mkdir(exist_ok=True)

            self.loop = asyncio.get_running_loop()
            self.session = aiohttp.ClientSession(timeout=session_timeout)

            await self.load_config_params()

            self.loop.create_task(scheduler(self))

            await asyncio.gather(
                self._message_handler(),
                self._message_polling(),
                self._run_scheduler()
            )

        except Exception as exc:
            if isinstance(self.session, ClientSession):
                await self.session.close()
                await asyncio.sleep(0.25)

            logging.exception(exc)

            await asyncio.sleep(60)

            await self.run()

    @async_elapsed_time
    async def load_config_params(self) -> None:
        logging.info('Loading params')

        with open(self.config_file, encoding='utf8') as config_file:
            with open(self.secrets_file) as secret_file:
                bot_config = json.loads(config_file.read())

                with open(self.commemorations_file) as comm_file:
                    self.commemorations = Commemorations(json.loads(comm_file.read()))

                with open(self.user_mood_file, encoding='utf8') as mood_file:
                    self.mood_per_user.update(json.loads(mood_file.read()))

                with open(self.user_opinions_file, encoding='utf8') as opinions_file:
                    self.user_opinions.update(json.loads(opinions_file.read()))

                bot_config.update(
                    json.loads(secret_file.read())
                )

                self.config = BotConfig(**bot_config)

                self.openai = OpenAiCompletion(
                    api_key=self.config.secrets.openai_key,
                    max_tokens=self.config.openai.max_tokens,
                    session=self.session,
                    semaphore=self.config.telegram_api_semaphore,
                    davinci_daily_limit=self.config.openai.davinci_daily_limit,
                    curie_daily_limit=self.config.openai.curie_daily_limit,
                    only_ada_users=self.config.openai.ada_only_users,
                    force_model=self.config.openai.force_model
                )

                self.allowed_list = [8375482, -704277411, -884201527, -20341310, -4098496372] if self.debug_mode else [
                    *[value.id for value in self.config.allowed_ids]]
                self.api_route = f"https://api.telegram.org/bot{self.config.secrets.bot_token}"

                self.faces_files = []
                self.alpha_faces_files = []
                self.faces_names = []
                self.face_embeddings = []

                self.semaphore = Semaphore(self.config.telegram_api_semaphore)

                for (_, _, filenames) in os.walk(self.face_images_path):
                    self.faces_files.extend(filenames)
                    break

                for (_, _, filenames) in os.walk(self.alpha_faces_path):
                    self.alpha_faces_files.extend(filenames)
                    break

                if not self.debug_mode:
                    for file in self.faces_files:
                        embeddings = face_recognition.face_encodings(
                            face_recognition.load_image_file(f"{self.face_images_path}/{file}")
                        )
                        if len(embeddings):
                            self.faces_names.append(file[:-7])
                            self.face_embeddings.append(embeddings[0])
                            logging.info(f"Loaded embeddings for {file}")
                        else:
                            logging.critical(f'NO EMBEDDINGS FOR {file}')

        logging.info('Loading chats')

        chats = os.listdir("chat_logs")
        self.chats_in_memory = defaultdict(list)
        for chat in chats:
            chat_dir = os.listdir(f"chat_logs/{chat}")
            for f in chat_dir:
                f_date = datetime.strptime(f.replace(".json", ""), "%Y-%m-%d")
                dif_days = (self.datetime_now - f_date).days
                if dif_days <= self.chat_in_memory_max_load_days:
                    with open(f"chat_logs/{chat}/{f}", "r") as chat_text:
                        json_chat = json.load(chat_text)
                        self.chats_in_memory[f"{chat}:{f.replace('.json','')}"] = json_chat

        self.mocked_today = False

        logging.info('Loading finished')

    async def _run_scheduler(self) -> None:
        while True:
            try:
                self.schedule.run_pending()
                await asyncio.sleep(self.polling_rate)

                if self.debug_mode:
                    logging.info(f'Scheduler is running. Total jobs: {len(self.schedule.get_jobs())}')
            except Exception as exc:
                self.loop.create_task(telegram_logging(exc))
                await asyncio.sleep(15)

    async def _message_polling(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.polling_rate)

                self.datetime_now = datetime.utcnow() - timedelta(hours=3)
                polling_url = f"{self.api_route}/getUpdates?offset={self.last_id}"

                async with self.session.get(polling_url) as request:
                    if 200 <= request.status < 300:
                        response = json.loads((await request.text()).replace('"from":{"', '"from_":{"'))
                        if 'ok' in response and response['ok']:
                            if self.debug_mode:
                                logging.info(f'Message polling task running:'
                                             f"{polling_url.replace(self.config.secrets.bot_token, '#TOKEN#')} last_id: {self.last_id + 1} - {self.datetime_now}")
                            self.messages = MessagesResults(**response)
                            self.last_id = self.messages.result[-1].update_id
            except Exception as exc:
                self.loop.create_task(telegram_logging(exc))
                await asyncio.sleep(15)

    async def _message_handler(self) -> None:
        while True:
            try:
                if self.debug_mode:
                    logging.info(f'Message controller task running - {len(self.interacted_updates)} - '
                                 f'Next roleta: {self.roleta_hour}')
                if hasattr(self.messages, 'result'):
                    for incoming_update in (entry for entry in self.messages.result
                                     if entry.update_id not in self.interacted_updates):

                        if self.debug_mode:
                            logging.info(incoming_update)

                        incoming_update: MessageReceived

                        if incoming_update is not None and incoming_update.message is not None:
                            chat_id = incoming_update.message.chat.id

                            self.loop.create_task(
                                self._store_messages_info(incoming_update)
                            )

                            self.messages_tasks[str(chat_id)].append(
                                self.loop.create_task(
                                    messages_coordinator(self, incoming_update)
                                )
                            )

                await asyncio.sleep(self.polling_rate)
            except Exception as exc:
                self.loop.create_task(telegram_logging(exc))
                await asyncio.sleep(15)

    async def _store_messages_info(self, incoming: MessageReceived):
        self.interacted_updates.append(incoming.update_id)

        if message := incoming.message:
            self.interacted_messages_with_chat_id.append(f"{message.chat.id}:"
                                                         f"{message.message_id}")

            if message.text is not None or message.caption is not None:
                date = str(self.datetime_now).split(' ')
                day_now = date[0]
                time_now = (date[-1].split(".")[0])[:-3]

                await asyncio.sleep(1)

                # todo refatorar isso quando tiver saco pelo amor de deus
                if message.caption:
                    self.messages_in_memory[message.chat.id].append(
                        f"{create_username(message.from_.first_name, message.from_.username)}: {message.caption[0:90]}")  # legacy
                    self.chats_in_memory[f"{message.chat.id}:{day_now}"].append(
                        f"{time_now} -"
                        f" {create_username(message.from_.first_name, message.from_.username)}: {message.caption[0:140]}")

                elif message.text:
                    if len(message.text) > 10:
                        self.messages_in_memory[message.chat.id].append(
                            f"{create_username(message.from_.first_name, message.from_.username)}: {message.text[0:90]}")  # legacy

                    self.chats_in_memory[f"{message.chat.id}:{day_now}"].append(
                        f"{time_now} -"
                        f" {create_username(message.from_.first_name, message.from_.username)}: {message.text[0:140]}")

    @async_elapsed_time
    async def image_downloader(
            self,
            message: TelegramMessage,
    ) -> T.Optional[bytes]:
        async with self.session.get(
                f"{self.api_route}/getFile?file_id={message.photo[-1].file_id}") as request:
            if 200 <= request.status < 300:
                response = json.loads(await request.text())
                if 'ok' in response and response['ok']:
                    file_path = response['result']['file_path']
                    async with self.session.get(f"{self.api_route.replace('.org/bot', '.org/file/bot')}/"
                                                f"{file_path}") as download_request:
                        if 200 <= download_request.status < 300:
                            return await download_request.read()
                        else:
                            logging.critical(f"Image download failed: {download_request.status}")

    async def send_photo(self, image: bytes, chat_id: int, caption=None, reply_to=None, sleep_time=0, max_retries=5) -> None:
        await asyncio.sleep(sleep_time)

        for _ in range(max_retries):
            try:
                async with self.semaphore:
                    async with self.session.post(
                            url=f"{self.api_route}/sendPhoto".replace('\n', ''),
                            data=aiohttp.FormData(
                                (
                                        ("chat_id", str(chat_id)),
                                        ("photo", image),
                                        ("reply_to_message_id", str(reply_to) if reply_to else ''),
                                        ('allow_sending_without_reply', 'true'),
                                        ("caption", caption if caption else '')
                                )
                            )
                    ) as resp:
                        logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")
                        if 200 <= resp.status < 300:
                            break
            except Exception as exc:
                self.loop.create_task(telegram_logging(exc))
            await asyncio.sleep(10)

    async def send_video(self, video: bytes, chat_id: int, reply_to=None, sleep_time=0) -> None:
        await asyncio.sleep(sleep_time)

        async with self.semaphore:
            async with self.session.post(
                    url=f"{self.api_route}/sendVideo".replace('\n', ''),
                    data=aiohttp.FormData(
                        (
                                ("chat_id", str(chat_id)),
                                ("video", video),
                                ("reply_to_message_id", str(reply_to) if reply_to else ''),
                                ('allow_sending_without_reply', 'true'),
                        )
                    )
            ) as resp:
                logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

    async def send_audio(self, audio: bytes, chat_id: int, reply_to=None, sleep_time=0) -> None:
        await asyncio.sleep(sleep_time)

        async with self.semaphore:
            async with self.session.post(
                    url=f"{self.api_route}/sendVoice".replace('\n', ''),
                    data=aiohttp.FormData(
                        (
                                ("chat_id", str(chat_id)),
                                ("voice", audio),
                                ("reply_to_message_id", str(reply_to) if reply_to else ''),
                                ('allow_sending_without_reply', 'true'),
                        )
                    )
            ) as resp:
                logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

    async def send_action(
            self,
            chat_id: int,
            action=T.Union[T.Literal['typing'], T.Literal['upload_photo'], T.Literal['find_location']],
            repeats=False
    ) -> None:
        while True:
            async with self.semaphore:
                async with self.session.post(
                        url=f"{self.api_route}/sendChatAction".replace('\n', ''),
                        data=aiohttp.FormData(
                            (
                                ("chat_id", str(chat_id)),
                                ('action', action),
                            )
                        )
                ) as resp:
                    logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

            if not repeats:
                break

            await asyncio.sleep(round(5 + (random.random() * 2)))

    async def send_document(self, document: bytes, chat_id: int, caption=None, reply_to=None, sleep_time=0) -> None:
        await asyncio.sleep(sleep_time)

        async with self.semaphore:
            async with self.session.post(
                    url=f"{self.api_route}/sendDocument".replace('\n', ''),
                    data=aiohttp.FormData(
                        (
                                ("chat_id", str(chat_id)),
                                ("document", document),
                                ("caption", caption if caption else ''),
                                ("reply_to_message_id", str(reply_to) if reply_to else ''),
                                ('allow_sending_without_reply', 'true'),
                        )
                    )
            ) as resp:
                logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

    async def forward_message(
            self,
            target_chat_id: int,
            from_chat_id: int,
            message_id: int,
            sleep_time=0,
            replace_token: T.Optional[str] = None
    ) -> int:
        await asyncio.sleep(sleep_time)
        url = self.api_route
        if replace_token:
            url = f"https://api.telegram.org/bot{replace_token}"

        async with self.semaphore:
            async with self.session.post(
                    url=f"{url}/forwardMessage".replace('\n', ''),
                    data=aiohttp.FormData(
                        (
                            ("chat_id", str(target_chat_id)),
                            ("from_chat_id", str(from_chat_id)),
                            ("message_id", str(message_id)),
                        )
                    )
            ) as resp:
                logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

                return resp.status

    async def send_message(
            self,
            message_text: str,
            chat_id: int,
            reply_to=None,
            sleep_time=0,
            parse_mode: str = "Markdown",
            disable_notification=False,
            max_retries=7,
            save_message=True
    ) -> None:
        fallback_parse_modes = ["", "HTML", "MarkdownV2", "Markdown"]

        await asyncio.sleep(sleep_time)

        for i in range(max_retries):
            if i == max_retries - 1:
                message_text = await send_message_last_try(message_text)
            async with self.semaphore:
                async with self.session.post(
                        f"{self.api_route}/sendMessage".replace('\n', ''),
                        json={
                            "chat_id": chat_id,
                            'reply_to_message_id': reply_to,
                            'allow_sending_without_reply': True,
                            'text': message_text,
                            'disable_notification': disable_notification,
                            'parse_mode': parse_mode
                        }
                ) as resp:
                    logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

                    if 200 <= resp.status < 300:
                        if save_message:
                            date = str(self.datetime_now).split(' ')
                            day_now = date[0]
                            time_now = (date[-1].split(".")[0])[:-3]

                            self.chats_in_memory[f"{chat_id}:{day_now}"].append(
                                f"{time_now} - Pedro: {message_text[0:150]}")

                            self.messages_in_memory[chat_id].append(
                                f"Pedro: {message_text[0:150]}")
                        break
                    parse_mode = fallback_parse_modes.pop() if len(fallback_parse_modes) else ""

    async def leave_chat(self, chat_id: int, sleep_time=0) -> None:
        await asyncio.sleep(sleep_time)

        async with self.session.post(
                f"{self.api_route}/leaveChat".replace('\n', ''),
                json={"chat_id": chat_id}
        ) as resp:
            logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        async with self.session.post(
                f"{self.api_route}/deleteMessage".replace('\n', ''),
                json={
                    "chat_id": chat_id,
                    "message_id": message_id
                }
        ) as resp:
            logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

    async def set_chat_title(self, chat_id: int, title: str) -> None:
        async with self.session.post(
                f"{self.api_route}/setChatTitle".replace('\n', ''),
                json={
                    "chat_id": chat_id,
                    "title": title
                }
        ) as resp:
            logging.info(f"{sys._getframe().f_code.co_name} - {resp.status}")

    async def is_taking_too_long(self, chat_id: int, user="", max_loops=2, timeout=20):
        if user:
            messages = [f"{user.lower()} ja vou te responder",
                        "meu cérebro tá devagar hoje",
                        f"só 1 minuto {user.lower()}"]

            for _ in range(max_loops):
                await asyncio.sleep(timeout + int(random.random() * timeout / 5))

                message = random.choice(messages)
                messages.remove(message)

                self.loop.create_task(
                    self.send_message(
                        message_text=message,
                        chat_id=chat_id
                    )
                )

                timeout *= 2

    @contextmanager
    def sending_action(
            self,
            chat_id: int,
            user="",
            action=T.Union[T.Literal['typing'], T.Literal['upload_photo'], T.Literal['find_location']]
    ):
        sending = self.loop.create_task(self.send_action(chat_id, action, True))
        timer = self.loop.create_task(self.is_taking_too_long(chat_id=chat_id, user=user))
        try:
            yield
        finally:
            sending.cancel()
            timer.cancel()


if __name__ == '__main__':
    pedro_leblon = FakePedro(
        bot_config_file='bot_configs.json',
        commemorations_file='commemorations.json',
        user_mood_file='user_mood.json',
        user_opinions_file='user_opinions.json',
        secrets_file=SECRETS_FILE,
        debug_mode=True,
    )

    asyncio.run(
        pedro_leblon.run()
    )
