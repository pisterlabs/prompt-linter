import json
import os
from datetime import timedelta
from io import BytesIO
from typing import List, Optional

import ffmpeg
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from loguru import logger
from pydantic import BaseModel

from ..events import message_handler, Context, CommandResult, Message, download_cmd, msg_cmd
from ..jid import parse_jid
from ..proto.wa_handler_pb2 import CommandResponse
from ..proto.wa_handler_pb2_grpc import ChatManagerStub
from ..store import ChatStore


class Change(BaseModel):
    change: str
    reason: str


class EnhancedTranscription(BaseModel):
    output: str
    changes: Optional[List[Change]]


@message_handler
def handle_message(ctx: Context, msg: Message) -> CommandResult:
    if msg.sent_before(timedelta(minutes=30)):
        return

    chat_jid, err = parse_jid(msg.chat)
    if err is not None:
        logger.warning(f"Failed to parse chat JID: {err}")
        return

    if chat_jid.is_group():
        store: ChatStore = ctx.store
        group = store.get_group(chat_jid)
        if group is None:
            logger.warning(f"Failed to get group {chat_jid}: {err}")
            return

        if not group.managed:
            logger.debug(f"Group {chat_jid} is not managed, but found a link. Ignoring.")
            return

    if msg.raw_message_event.info.type != "media" or msg.raw_message_event.info.media_type != "ptt":
        return

    if msg.raw_message_event.message.audioMessage is None or msg.raw_message_event.message.audioMessage.url is None or \
            msg.raw_message_event.message.audioMessage.url == "":
        logger.warning(f"Invalid audio message: {msg.raw_message_event}")
        return

    stub: ChatManagerStub = ctx.stub
    resp: CommandResponse = stub.Execute(download_cmd(msg.raw_message_event))
    if resp.download_response is None:
        logger.warning(f"Failed to download message {msg.raw_message_event.info.id}: {resp.error}")
        return

    if resp.download_response.message_id != msg.raw_message_event.info.id or resp.download_response.data is None:
        logger.warning(f"Failed to download message {msg.raw_message_event.info.id}: {resp.error}")
        return

    try:
        mp3bytes, _ = (
            ffmpeg.input("pipe:", format="ogg", acodec="libopus")
            .output("-", format="mp3", acodec='libmp3lame', ac=1, ar=16000, ab="32k",
                    af="asendcmd=c='0.0 afftdn@n sn start; 1.0 afftdn@n sn stop',afftdn@n")
            .run(
                cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, input=resp.download_response.data
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    mp3 = BytesIO(mp3bytes)
    mp3.name = f"{resp.download_response.message_id}.mp3"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    transcription = openai.Audio.transcribe("whisper-1", mp3)
    if transcription.text is None or transcription.text == "":
        logger.warning(f"Failed to transcribe audio: {transcription}")
        return

    if len(transcription.text.split(" ")) < 4:
        yield msg_cmd(msg.chat, f"@{msg.sender_jid.user} said:\n{transcription.text}", reply_to=msg.raw_message_event)
        return

    llm = ChatOpenAI(temperature=0)
    prompt = """
The following message might have transcription errors wrapped with quotes. If it has transcription errors, fix them and state your changes and reasons.

Respond in a JSONL format:  changes<[]change<{reason<str>, change<str>}>, output<str>  where each change should include a 'change' field containing the corrected word and a 'reason' field explaining why the change was made. If you don't have any changes, omit the changes field.

Respond ONLY in a valid compacted JSONL format, do not add comments. DO NOT reply to the content, alter its meaning or answer questions present in the content. Only fix transcriptions error if present as instructed above.
"""
    result: AIMessage = llm([
        SystemMessage(content=prompt),
        HumanMessage(content='"היי בוט, מה השעה?"'),
        AIMessage(content='{"changes": [], "output": "היי בוט, מה השעה?"}'),
        HumanMessage(content='"Welcome abroad Daniel, it\'s so cruel to have you here! :)"'),
        AIMessage(
            content="""{"changes": [{"reason": "Changed 'abroad' to 'aboard' because it makes more sense in the context of welcoming someone onto a ship or plane.","change": "abroad"}, {"reason": "Changed 'cruel' to 'cool' because it makes more sense in the context of welcoming someone and expressing excitement about their presence.","change": "cruel"}], "output": "Welcome aboard Daniel, it's so cool to have you here! :)"}"""),
        HumanMessage(content=json.dumps(transcription.text)),
    ])

    if result.content is None or result.content == "":
        logger.warning(f"Failed to enhance transcription: {result}")
        yield msg_cmd(msg.chat, f"@{msg.sender_jid.user} said:\n{transcription.text}", reply_to=msg.raw_message_event)
        return

    try:
        et = EnhancedTranscription.parse_raw(result.content)
    except Exception as e:
        logger.warning(f"Failed to parse enhanced transcription: {e}")
        yield msg_cmd(msg.chat, f"@{msg.sender_jid.user} said:\n{transcription.text}", reply_to=msg.raw_message_event)
        return

    if et.output is None or et.output == "":
        logger.warning(f"Failed to enhance transcription: {result}")
        yield msg_cmd(msg.chat, f"@{msg.sender_jid.user} said:\n{transcription.text}", reply_to=msg.raw_message_event)
        return

    yield msg_cmd(msg.chat, f"@{msg.sender_jid.user} said:\n{et.output}", reply_to=msg.raw_message_event)
