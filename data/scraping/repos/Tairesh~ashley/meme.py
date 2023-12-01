import json
import os
import random
import re
from typing import List
from urllib.parse import quote

import openai
import requests
from telebot.types import Message

from ashlee import emoji, utils, pepe, stickers
from ashlee.action import Action
from ashlee.utils import unique


class Meme(Action):

    r_word = re.compile(r"[\w\d\-\']+", flags=re.IGNORECASE)
    PIXABAI_API = "https://pixabay.com/api/?key=%KEY%&orientation=horizontal&min_width=700&min_height=500&q={}"

    def get_description(self) -> str:
        return "случайно сгенерированный мем"

    def get_name(self) -> str:
        return emoji.FUN + " Meme"

    def get_cmds(self) -> List[str]:
        return ["meme", "mem"]

    def get_keywords(self) -> List[str]:
        return ["сделай мем", "скинь мем", "сгенерируй мем"]

    def after_loaded(self):
        Meme.PIXABAI_API = Meme.PIXABAI_API.replace(
            "%KEY%", self.tgb.api_keys["pixabay_apikey"]
        )

    @Action.save_data
    @Action.send_uploading_photo
    def call(self, message: Message):

        file_url = None
        if message.reply_to_message:
            if message.reply_to_message.content_type == "photo":
                file_id = message.reply_to_message.photo.pop().file_id
                file = self.bot.get_file(file_id=file_id)
                file_url = (
                    f"https://api.telegram.org/file/bot{self.tgb.token}/"
                    + file.file_path
                )
                if message.text.startswith("/"):
                    keyword = utils.get_keyword(message)
                    if keyword:
                        text = keyword
                    else:
                        text = message.reply_to_message.caption
                else:
                    text = message.text
            else:
                text = message.reply_to_message.text
        elif message.text.startswith("/"):
            text = utils.get_keyword(message)
        else:
            text = message.text
            if " мем про " in text:
                text = text.split(" мем про ")[1]
            else:
                for k in self.get_keywords():
                    text = text.replace(k, "")

        tries = 0
        while tries < 10:
            tries += 1
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Придумай смешную подпись к мему на тему {text} без кавычек не длинее четырёх слов",
                temperature=1,
                max_tokens=30,
            )
            sentence = response["choices"][0]["text"]
            for ch in (",", ".", "..", "...", "*"):
                if ch in sentence:
                    sentence = sentence.split(ch)[0]
            words = unique(sentence.split(" "))
            random.shuffle(words)

            if file_url is None:
                urls = []
                for word in words:
                    for subword in self.r_word.findall(word):
                        try:
                            data = json.loads(
                                requests.get(
                                    self.PIXABAI_API.format(quote(subword))
                                ).content.decode("utf-8")
                            )
                            if data["totalHits"] > 0:
                                for hit in data["hits"]:
                                    urls.append(hit["largeImageURL"])
                        except json.decoder.JSONDecodeError:
                            continue
            else:
                urls = [file_url]
            random.shuffle(urls)
            for url in urls:
                try:
                    file_name = utils.download_file(url)
                    try:
                        pepe.memetize(file_name, sentence)
                        self.bot.send_photo(
                            message.chat.id,
                            open(file_name, "rb"),
                            reply_to_message_id=message.message_id,
                        )
                        os.remove(file_name)
                        return
                    except Exception:
                        os.remove(file_name)
                except Exception:
                    pass

        self.bot.send_sticker(
            message.chat.id, stickers.FOUND_NOTHING, message.message_id
        )
