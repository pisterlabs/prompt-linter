import botpy
from botpy.message import Message
import openai

from bot.mdb import QQ_MESSAGE_TABLE
from bot.midjourney import MidjourneyAPI
from config.env import APP_ID, APP_TOKEN

mj_sdk = MidjourneyAPI()

BOOT_AT_TEXT = "<@!6326807383311494599>"

SUPPORT_COMMANDS = ('U1', 'U2', 'U3', 'U4', 'V1', 'V2', 'V3', 'V4', 'R')
COMMAND_INDEX = {
    "U1": 1,
    "U2": 2,
    "U3": 3,
    "U4": 4,
    "V1": 1,
    "V2": 2,
    "V3": 3,
    "V4": 4
}


class MyClient(botpy.Client):

    @staticmethod
    def build_custom_id(msg_hash, prompt):
        if "R" == prompt:
            return f"MJ::JOB::variation::1::{msg_hash}::SOLO"
        index = prompt[-1]
        if "U" in prompt:
            return f"MJ::JOB::upsample::{index}::{msg_hash}"
        if "V" in prompt:
            return f"MJ::JOB::variation::{index}::{msg_hash}"
        return None

    async def on_at_message_create(self, message: Message):
        print(message)
        message_id = message.id

        QQ_MESSAGE_TABLE.update_one({"qq_id": message_id}, {"$set": {
            "qq_message": {
                "author": str(message.author),
                "content": message.content,
                "channel_id": message.channel_id,
                "id": message.id,
                "guild_id": message.guild_id,
                "member": str(message.member),
                "message_reference": str(message.message_reference),
                "mentions": str(message.mentions),
                "attachments": str(message.attachments),
                "seq": message.seq,
                "seq_in_channel": message.seq_in_channel,
                "timestamp": message.timestamp,
                "event_id": message.event_id
            }
        }}, upsert=True)

        content = message.content
        prompt = content.replace(BOOT_AT_TEXT, "").strip()

        if content.startswith(BOOT_AT_TEXT) and '/会话' in content:
            prompt = prompt.replace("/会话", "").strip()
            if not prompt:
                await message.reply(content="请输入聊天内容", message_reference={"message_id": message_id})
                return
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                    {"role": "user", "content": prompt},
                ]
            )
            await message.reply(content=response['choices'][0]['message']['content'],
                                message_reference={"message_id": message_id})
            return

        if content.startswith(BOOT_AT_TEXT) and '/绘图' in content:
            prompt = prompt.replace("/绘图", "").strip()
            res = mj_sdk.create_imagine(prompt=prompt, message_id=message_id)
            if res and isinstance(res, bool):
                await message.reply(content=f"排队中了~\n任务ID: {message_id}",
                                    message_reference={"message_id": message_id})
                return
            await message.reply(content=res, message_reference={"message_id": message_id})
            return

        message_reference_id = message.message_reference.message_id
        qq_log = await self.api.get_message(channel_id=message.channel_id, message_id=message_reference_id)
        content = qq_log.get("message").get("content")
        _ids = content.split("  ")[-1].strip()
        old_qq_id = _ids.split("::")[0].strip()
        old_mj_id = _ids.split("::")[1].strip()
        log = QQ_MESSAGE_TABLE.find_one({"qq_id": old_qq_id, "mj_id": old_mj_id})
        if prompt in SUPPORT_COMMANDS and log:
            commands = log.get("commands") or []
            if prompt in commands and "U" in prompt:
                await message.reply(content=f"已经处理过了: {message_id}",
                                    message_reference={"message_id": message_id})
                return

            commands.append(prompt)
            QQ_MESSAGE_TABLE.update_one({"qq_id": message_id}, {"$set": {
                "commands": commands
            }})

            mj_message = log.get("mj_message")
            msg_hash = mj_message.get("msgHash")
            custom_id = self.build_custom_id(msg_hash, prompt)
            if not custom_id:
                await message.reply(content=f"存在异常\n任务ID: {message_id} \ncustom_id生成失败",
                                    message_reference={"message_id": message_id})
                return
            res = mj_sdk.up_imagine(mj_message_id=mj_message.get("messageId"), custom_id=custom_id)
            if res and isinstance(res, bool):
                await message.reply(content=f"排队中了~\n任务ID: {message_id}",
                                    message_reference={"message_id": message_id})
                return
            await message.reply(content=res, message_reference={"message_id": message_id})
            return
        await message.reply(content=f"不支持的命令\n支持的命令: {','.join(SUPPORT_COMMANDS)}",
                            message_reference={"message_id": message_id})


def bot_start():
    intents = botpy.Intents(public_guild_messages=True, guild_messages=True)
    client = MyClient(intents=intents)
    client.run(appid=f"{APP_ID}", token=f"{APP_TOKEN}")
