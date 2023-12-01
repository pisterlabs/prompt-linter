from discord.ext import commands
import base64
import json
from dotenv import load_dotenv
import openai
import os
import asyncio

load_dotenv()
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
openai.api_key = OPENAI_TOKEN
channel_id = 1017700772466663487


class Message(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    conversation = []
    isDiscussion = False

    async def getChatResponse():
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=Message.conversation,
        )
        Message.conversation.append(
            {
                "role": "user",
                "content": response["choices"][0]["message"]["content"],
            }
        )
        return response

    def resetConversation(tone, topic):
        Message.conversation = [
            {
                "role": "system",
                "content": f"You are a chatbot, you are going to debate with a given topic, you are free to choose the stance and convince me. your tone should be {tone}. The topic to be debated is {topic}. Limit your response to less than 40 words each time.",
            }
        ]

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return
        if message.content.startswith("[SYN]"):
            payload = message.content[len("[SYN]") :]
            decoded_payload = base64.b64decode(payload).decode("utf-8")
            data = json.loads(decoded_payload)
            Message.resetConversation(data["tone"], data["topic"])
            Message.isDiscussion = True
            response = await Message.getChatResponse()
            channel = self.bot.get_channel(channel_id)
            await channel.send(
                f"{response['choices'][0]['message']['content']}",
                silent=True,
            )
            return
        if Message.isDiscussion:
            if message.content.startswith("[DISCUSSION END]"):
                Message.isDiscussion = False
                return
            elif Message.isDiscussion:
                await asyncio.sleep(20)
                Message.conversation.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
                response = await Message.getChatResponse()
                channel = self.bot.get_channel(channel_id)
                if response["usage"]["total_tokens"] > 1000:
                    await channel.send(
                        "[DISCUSSION END]",
                        silent=True,
                    )
                    return

                await channel.send(
                    f"{response['choices'][0]['message']['content']}",
                    silent=True,
                )


def setup(bot):
    bot.add_cog(Message(bot))
