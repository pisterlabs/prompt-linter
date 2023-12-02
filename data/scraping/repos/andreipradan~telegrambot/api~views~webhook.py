import logging
import os
import random
from datetime import datetime
from time import sleep

import openai
import pytz
import requests
import telegram

from flask import Blueprint
from flask import request
from telegram import error

from core import handlers, database
from core import local_data
from core import utils
from core.constants import GAME_COMMANDS
from core.constants import GOOGLE_CLOUD_COMMANDS
from core.constants import GOOGLE_CLOUD_WHITELIST
from inlines import inline
from powers.analyze_sentiment import analyze_sentiment
from powers.games import Games
from powers.translate import translate_text
from scrapers.formatters import parse_global

webhook_views = Blueprint("webhook_views", __name__)

TOKEN = os.environ["TOKEN"]


logging.basicConfig(
    format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@webhook_views.route(f"/webhook/{TOKEN}", methods=["POST"])
def webhook():
    json = request.get_json()
    if not json:
        raise ValueError("No payload")

    bot = telegram.Bot(token=TOKEN)
    update = telegram.Update.de_json(json, bot)

    command_text, status_code = handlers.validate_components(update)

    if command_text == "inline":
        chat_id = update.callback_query.message.chat_id
        if status_code in ["more", "back", "end"]:
            return getattr(inline, status_code)(update)
        if status_code.startswith("games_"):
            status_code = status_code.replace("games_", "")
            return inline.refresh_data(
                update, Games(chat_id, status_code).get()
            )
        return inline.refresh_data(update, getattr(local_data, status_code)())

    if status_code == 1337:
        if command_text == "skip-debug":
            return "ok"
        text = f"{command_text}.\nUpdate: {update.to_dict()}"
        return utils.send_message(bot, text=text)

    who = update.message.from_user.username or update.message.from_user.id
    if str(who) not in os.getenv("WHITELIST", "").split(","):
        logging.error(f"Ignoring message from: {who}")
        return ""

    chat_id = update.message.chat.id
    if status_code != "valid-command":
        return utils.send_message(bot, text=command_text, chat_id=chat_id)

    if command_text == "start":
        return inline.start(update)

    if command_text == "games" and not update.message.text.split(" ")[1:]:
        return inline.start(update, games=True)

    if command_text in GOOGLE_CLOUD_COMMANDS:
        chat_type = update.message.chat.type
        if str(chat_id) not in GOOGLE_CLOUD_WHITELIST[chat_type]:
            return utils.send_message(bot, "Unauthorized", chat_id)

        arg = " ".join(update.message.text.split(" ")[1:])
        if command_text == "translate":
            return utils.send_message(bot, translate_text(arg), chat_id)
        if command_text == "analyze_sentiment":
            return utils.send_message(bot, analyze_sentiment(arg), chat_id)

    if command_text in GAME_COMMANDS:
        chat_type = update.message.chat.type
        if str(chat_id) not in GOOGLE_CLOUD_WHITELIST[chat_type]:
            return utils.send_message(bot, "Unauthorized", chat_id)

        if command_text == "games":
            args = update.message.text.split(" ")[1:]
            if len(args) not in (2, 3) or len(args) == 2 and args[1] != "new":
                return utils.send_message(
                    bot,
                    parse_global(
                        title="Syntax",
                        stats=[
                            "/games => scores",
                            "/games <game_name> new => new game",
                            "/games <game_name> new_player <player_name>"
                            " => new player",
                            "/games <game_name> + <player_name>"
                            " => increase <player_name>'s score by 1",
                            "/games <game_name> - <player_name>"
                            " => decrease <player_name>'s score by 1",
                        ],
                        items={},
                    ),
                    chat_id,
                )
            name, *args = args
            games = Games(chat_id, name)
            if len(args) == 1:
                return utils.send_message(bot, games.new_game(), chat_id)
            return utils.send_message(bot, games.update(*args), chat_id)

        if command_text == "randomize":
            args = update.message.text.split(" ")[1:]
            if len(args) not in range(2, 51):
                return utils.send_message(
                    bot,
                    "Must contain a list of 2-50 items separated by space",
                    chat_id,
                )
            random.shuffle(args)
            return utils.send_message(
                bot,
                "\n".join(f"{i+1}. {item}" for i, item in enumerate(args)),
                chat_id,
            )
    if command_text.startswith("save"):

        def save_to_db(message, text_override=None):
            author = message.from_user.to_dict()
            author["full_name"] = message.from_user.full_name
            database.get_collection("saved-messages").insert_one(
                {
                    "author": author,
                    "chat_id": message.chat_id,
                    "chat_name": message.chat.title,
                    "date": message.date,
                    "message": {
                        "id": message.message_id,
                        "text": text_override or message.text,
                    },
                    "saved_at": update.message.date.utcnow(),
                    "saved_by": update.message.from_user.to_dict(),
                }
            )

        if command_text == "save-group-name":
            chat_description = bot.get_chat(update.message.chat_id).description
            save_to_db(update.message, text_override=chat_description)
            return update.message.reply_text(
                text="Saved ✔",
                disable_notification=True,
            ).to_json()
        if command_text == "save":
            if not update.message.reply_to_message:
                return update.message.reply_text(
                    text="This command must be sent "
                    "as a reply to the message you want to save",
                    disable_notification=True,
                    disable_web_page_preview=True,
                    parse_mode=telegram.ParseMode.HTML,
                ).to_json()
            if not (
                update.message.reply_to_message.text
                or update.message.reply_to_message.new_chat_title
            ):
                return update.message.reply_text(
                    text="No text found to save.",
                    disable_notification=True,
                    disable_web_page_preview=True,
                    parse_mode=telegram.ParseMode.HTML,
                ).to_json()

            save_to_db(
                update.message.reply_to_message,
                text_override=bot.get_chat(update.message.chat_id).description
                if update.message.reply_to_message.new_chat_title
                else None,
            )
            return update.message.reply_text(
                text="Saved ✔",
                disable_notification=True,
            ).to_json()
        if command_text == "saved":
            items = list(
                database.get_many(
                    collection="saved-messages",
                    order_by="date",
                    how=1,
                    chat_id=chat_id,
                )
            )
            if not items:
                return update.message.reply_text(
                    text="No saved messages in this chat.",
                    disable_notification=True,
                ).to_json()

            def link(item):
                return f"""
*{item['chat_name']}*
- de {item['author']['full_name']}, {item['date'].strftime("%d %b %Y %H:%M")}

{item['message']['text']} ([link](https://t.me/c/{str(chat_id)[3:]}/{item['message']['id']}))"""

            messages = []
            message = ""
            logger.info(f"Got {len(items)} saved messages")
            for item in items:
                current_item = f"\n\n{link(item)}"
                if len(message) + len(current_item) > 4000:
                    messages.append(message)
                    message = ""
                message += current_item

            if message not in messages:
                messages.append(message)

            logger.debug(f"Sending {len(messages)} telegram messages")
            for msg in messages:
                sleep(1)
                return utils.send_message(
                    bot,
                    text=msg,
                    chat_id=chat_id,
                )
            logger.debug("Done")

    if command_text == "get_chat_id":
        return update.message.reply_text(
            text=f"Chat ID: {update.message.chat_id}",
            disable_notification=True,
        ).to_json()

    if command_text == "chatgpt":
        openai.api_key = os.environ["GPT3_API_KEY"]
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=update.message.text,
                max_tokens=1024,
            )
            text = response.choices[0].text
            try:
                return update.message.reply_text(
                    text, disable_notification=True
                ).to_json()
            except error.BadRequest as e:
                logging.warning(
                    f"Bad request: {e}. Trying to split the message"
                )
                half = int(len(text) / 2)
                update.message.reply_text(
                    text=text[:half] + " [[1/2 - continued below..]]",
                    disable_notification=True,
                )
                return update.message.reply_text(
                    text="[[2/2]] " + text[half:], disable_notification=True
                )

        except Exception as e:
            logging.exception(e)
            return update.message.reply_text(str(e)).to_json()

    if command_text == "bus":
        try:
            args = update.message.text.split(" ")[1:]
        except Exception as e:
            return update.message.reply_text(
                f"Error: {e.args}",
                parse_mode=telegram.ParseMode.HTML,
                disable_notification=True,
            ).to_json()

        if len(args) != 1:
            return update.message.reply_text(
                f"Only 1 bus number allowed, got: {len(args)}",
                parse_mode=telegram.ParseMode.HTML,
                disable_notification=True,
            ).to_json()

        bus_number = args[0]
        if not bus_number.isnumeric():
            return update.message.reply_text(
                f"Invalid number: {bus_number}",
                parse_mode=telegram.ParseMode.HTML,
                disable_notification=True,
            ).to_json()

        now = datetime.now(pytz.timezone("Europe/Bucharest"))
        weekday = now.weekday()
        if weekday in range(5):
            day = "lv"
        elif weekday == 5:
            day = "s"
        elif weekday == 6:
            day = "d"
        else:
            logger.error("This shouldn't happen, like ever")
            return ""

        headers = {"Referer": "https://ctpcj.ro/"}
        resp = requests.get(
            f"https://ctpcj.ro/orare/csv/orar_{bus_number}_{day}.csv",
            headers=headers,
        )
        if resp.status_code != 200 or "EROARE" in resp.text:
            return update.message.reply_text(
                text=f"Bus {bus_number} not found",
                parse_mode=telegram.ParseMode.HTML,
                disable_notification=True,
            ).to_json()

        lines = [l.strip() for l in resp.text.split("\n")]
        route = lines.pop(0).split(",")[1]
        days_of_week = lines.pop(0).split(",")[1]
        date_start = lines.pop(0).split(",")[1]
        lines.pop(0)  # start station
        lines.pop(0)  # stop station

        current_bus_index = None
        current_bus = lines[0]
        for i, bus in enumerate(lines):
            start, *stop = bus.split(",")
            now_time = now.strftime("%H:%M")
            if start > now_time or (stop and stop[0] > now_time):
                current_bus_index = i
                current_bus = lines[i]
                break

        if current_bus_index:
            lines = (
                lines[current_bus_index - 3 : current_bus_index]
                + lines[current_bus_index : current_bus_index + 3]
            )

        all_rides = "\n".join(lines)
        text = f"Next <b>{bus_number}</b> at {current_bus}\n\n{route}\n{days_of_week}:\n(Available from: {date_start}) \n{all_rides}"
        return update.message.reply_text(
            text=text,
            parse_mode=telegram.ParseMode.HTML,
            disable_notification=True,
        ).to_json()
    raise ValueError(f"Unhandled command: {command_text}, {status_code}")
