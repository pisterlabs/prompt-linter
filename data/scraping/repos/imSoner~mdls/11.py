__version__ = (1, 0, 1)

import contextlib


from telethon.tl.types import Message
import requests
import logging
import re

from .. import loader, utils

logger = logging.getLogger(__name__)


@loader.tds
class ChatGPT(loader.Module):

    strings = {
        "name": "ChatGPT",
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>No arguments"
            " provided</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Question:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Answer:</b> {answer}"
        ),
        "loading": "<code>Loading...</code>",
        "no_api_key": (
            "<b>🚫 No API key provided</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Get it from official OpenAI"
            " website and add it to config</i>"
        ),
    }

    strings_ru = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>Не указані"
            " аргументи</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Питання:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Відповідь:</b> {answer}"
        ),
        "loading": "<code>Загрузка...</code>",
        "no_api_key": (
            "<b>🚫 Не указан API ключ</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Отримайте його на офіційному"
            " сайті OpenAI і додайте в конфіг</i>"
        ),
    }

    strings_es = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>No se han"
            " proporcionado argumentos</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Pregunta:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Respuesta:</b>"
            " {answer}"
        ),
        "loading": "<code>Cargando...</code>",
        "no_api_key": (
            "<b>🚫 No se ha proporcionado una clave API</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Obtenga una en el sitio web"
            " oficial de OpenAI y agréguela a la configuración</i>"
        ),
    }

    strings_fr = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>Aucun argument"
            " fourni</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Question:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Réponse:</b> {answer}"
        ),
        "loading": "<code>Chargement...</code>",
        "no_api_key": (
            "<b>🚫 Aucune clé API fournie</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Obtenez-en un sur le site"
            " officiel d'OpenAI et ajoutez-le à la configuration</i>"
        ),
    }

    strings_de = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>Keine Argumente"
            " angegeben</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Frage:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Antwort:</b> {answer}"
        ),
        "loading": "<code>Laden...</code>",
        "no_api_key": (
            "<b>🚫 Kein API-Schlüssel angegeben</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Holen Sie sich einen auf der"
            " offiziellen OpenAI-Website und fügen Sie ihn der Konfiguration hinzu</i>"
        ),
    }

    strings_tr = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>Argümanlar"
            " verilmedi</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Soru:</b> {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Cevap:</b> {answer}"
        ),
        "loading": "<code>Yükleniyor...</code>",
        "no_api_key": (
            "<b>🚫 API anahtarı verilmedi</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> OpenAI'nın resmi websitesinden"
            " alın ve yapılandırmaya ekleyin</i>"
        ),
    }

    strings_uz = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>Argumentlar"
            " ko'rsatilmadi</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Savol:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Javob:</b> {answer}"
        ),
        "loading": "<code>Yuklanmoqda...</code>",
        "no_api_key": (
            "<b>🚫 API kalit ko'rsatilmadi</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Ofitsial OpenAI veb-saytidan"
            " oling</i>"
        ),
    }

    strings_it = {
        "no_args": (
            "<emoji document_id=5312526098750252863>🚫</emoji> <b>Nessun argomento"
            " fornito</b>"
        ),
        "question": (
            "<emoji document_id=5974038293120027938>👤</emoji> <b>Domanda:</b>"
            " {question}\n"
        ),
        "answer": (
            "<emoji document_id=5199682846729449178>🤖</emoji> <b>Risposta:</b> {answer}"
        ),
        "loading": "<code>Caricamento...</code>",
        "no_api_key": (
            "<b>🚫 Nessuna chiave API fornita</b>\n<i><emoji"
            " document_id=5199682846729449178>ℹ️</emoji> Ottienila dal sito ufficiale"
            " di OpenAI e aggiungila al tuo file di configurazione</i>"
        ),
    }

    def __init__(self):
        self.config = loader.ModuleConfig(
            loader.ConfigValue(
                "api_key",
                "",
                "API key from OpenAI",
                validator=loader.validators.Hidden(loader.validators.String()),
            ),
        )

    async def _make_request(
            self,
            method: str,
            url: str,
            headers: dict,
            data: dict,
    ) -> dict:
        resp = await utils.run_sync(
            requests.request,
            method,
            url,
            headers=headers,
            json=data,
        )
        return resp.json()

    def _process_code_tags(self, text: str) -> str:
        return re.sub(
            r"`(.*?)`",
            r"<code>\1</code>",
            re.sub(r"```(.*?)```", r"<code>\1</code>", text, flags=re.DOTALL),
            flags=re.DOTALL,
        )

    async def _get_chat_completion(self, prompt: str) -> str:
        resp = await self._make_request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f'Bearer {self.config["api_key"]}',
            },
            data={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.get("error", None):
            return f"🚫 {resp['error']['message']}"
        return resp["choices"][0]["message"]["content"]

    @loader.command(
        ru_doc="<питання> - Спитати",
        it_doc="<domanda> - Fai una domanda",
        fr_doc="<question> - Posez une question",
        de_doc="<frage> - Stelle eine Frage",
        es_doc="<pregunta> - Haz una pregunta",
        tr_doc="<soru> - Soru sor",
        uz_doc="<savol> - Savol ber",
    )
    async def gpt(self, message: Message):
        """<question> - Ask a question"""
        if self.config["api_key"] == "":
            return await utils.answer(message, self.strings("no_api_key"))

        args = utils.get_args_raw(message)
        if not args:
            return await utils.answer(message, self.strings("no_args"))

        await utils.answer(
            message,
            "\n".join(
                [
                    self.strings("question").format(question=args),
                    self.strings("answer").format(answer=self.strings("loading")),
                ]
            ),
        )
        answer = await self._get_chat_completion(args)
        await utils.answer(
            message,
            "\n".join(
                [
                    self.strings("question").format(question=args),
                    self.strings("answer").format(
                        answer=self._process_code_tags(answer)
                    ),
                ]
            ),
        )