import sys
import traceback
import warnings
from contextlib import asynccontextmanager

import openai
from discord import File, Interaction, Message
from discord import errors as discord_errors
from discord.abc import Messageable
from discord.app_commands import errors as app_cmd_errors
from discord.ext.commands import errors as ext_cmd_errors
from loguru import logger

from chatbot.utils.datetime import utcnow
from chatbot.utils.discord import Color2, Embed2
from chatbot.utils.discord.file import discord_open
from chatbot.utils.discord.typing import OutgoingMessage


def system_message():
    return Embed2().set_footer("System message")


def is_system_message(message: Message):
    return any(
        embed.footer and embed.footer.text == "System message"
        for embed in message.embeds
    )


@asynccontextmanager
async def report_warnings(messageable: Messageable):
    from chatbot.utils.discord.ui import ErrorReportView

    with warnings.catch_warnings(record=True) as messages:
        try:
            yield
        finally:
            if messages:
                report = (
                    system_message()
                    .set_color(Color2.orange())
                    .set_title("Warnings")
                    .set_description("\n".join([str(m.message) for m in messages]))
                )
                with logger.catch(Exception):
                    await messageable.send(embed=report, view=ErrorReportView())


async def report_error(
    error: Exception,
    *,
    interaction: Interaction | None = None,
    messageable: Messageable | None = None,
):
    from chatbot.utils.discord.ui import ErrorReportView

    error = getattr(error, "original", None) or error.__cause__ or error

    def get_traceback() -> File:
        if not isinstance(error, BaseException):
            return
        tb = traceback.format_exception(error)
        tb_body = "".join(tb)
        for path in sys.path:
            tb_body = tb_body.replace(path, "")
        filename = f'stacktrace.{utcnow().isoformat().replace(":", ".")}.py'
        with discord_open(filename) as (stream, file):
            stream.write(tb_body.encode())
        return file

    tb = None

    match error:
        case (
            ext_cmd_errors.UserInputError()
            | ext_cmd_errors.CheckFailure()
            | app_cmd_errors.CheckFailure()
            | openai.InvalidRequestError()
        ):
            color = Color2.orange()
            title = "HTTP 400 Bad Request"
        case discord_errors.Forbidden():
            color = Color2.red()
            title = "HTTP 403 Forbidden"
        case discord_errors.NotFound():
            color = Color2.dark_gray()
            title = "HTTP 404 Forbidden"
        case (
            ext_cmd_errors.CommandOnCooldown()
            | ext_cmd_errors.MaxConcurrencyReached()
            | ext_cmd_errors.BotMissingPermissions()
            | app_cmd_errors.BotMissingPermissions()
        ):
            color = Color2.dark_red()
            title = "HTTP 503 Service Unavailable"
        case ext_cmd_errors.CommandNotFound():
            return
        case _:
            color = Color2.red()
            title = "HTTP 500 Internal Server Error"
            logger.exception(error)
            tb = get_traceback()

    if not (interaction or messageable):
        return

    report = (
        system_message().set_color(color).set_title(title).set_description(str(error))
    )

    with logger.catch(Exception):
        response: OutgoingMessage = {"embeds": [report]}
        if tb:
            response["files"] = [tb]
        if interaction:
            response["ephemeral"] = True
            if interaction.response.is_done():
                await interaction.followup.send(**response)
            else:
                await interaction.response.send_message(**response)
        if messageable:
            response["view"] = ErrorReportView()
            await messageable.send(**response)


async def unbound_error_handler(interaction: Interaction, error: Exception):
    return await report_error(error, interaction=interaction)
