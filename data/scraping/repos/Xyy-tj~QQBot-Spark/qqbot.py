import botpy
from botpy.message import Message
import openai
import sqlite3
import _thread as thread
import ssl
from datetime import datetime
from spark_gpt.spark_gpt import SparkGPT
from botpy.message import Message, DirectMessage
from botpy.interaction import Interaction


def connect_sqlite():
    conn = sqlite3.connect('qqbot.db')
    cursor = conn.cursor()
    return

def spark_demo(question="你好"):
    speaker = SparkGPT("你是一个QQ频道机器人，名字是猫猫，请根据用户指令加以回复。", language="chinese")
    answer = speaker.ask(question)
    return answer

def send_img():
    img = "http://zyfan.zone/wp-content/uploads/2023/07/1689743930948.png"
    return img

class MyClient(botpy.Client):
    async def on_at_message_create(self, message: Message):
        # 被@时回复
        # await self.api.post_message(channel_id=message.channel_id, content=spark_demo(message.content))
        await self.api.post_message(channel_id=message.channel_id, content=spark_demo(message.content), image=send_img())
        
    async def on_direct_message_create(self, message: DirectMessage):
        await self.api.post_dms(
            guild_id=message.guild_id,
            content=f"机器人{self.robot.name}收到你的私信了: {message.content}",
            msg_id=message.id,
        )


intents = botpy.Intents(public_guild_messages=True, direct_message=True, interaction=True) 
client = MyClient(intents=intents)
client.run(appid="102069399", token="fku0DRitB6XmEOdBBeJM2ibWdzkGdA0Q")
