from Global import *
import discord
import tiktoken
from langchain.callbacks import get_openai_callback
import json


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def period(delta, pattern):
    d = {'d': delta.days}
    d['h'], rem = divmod(delta.seconds, 3600)
    d['m'], d['s'] = divmod(rem, 60)
    return pattern.format(**d)


class OpenAIBot(discord.Client):
    def __init__(self):
        self.discord_bot_token = os.environ.get('discord_bot_token')
        super().__init__(intents=discord.Intents.all())

        self.token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0
        }
        self.token_path = os.path.join(
            os.path.dirname(__file__), "token_count.csv")

        self.save_token_Usage("read")
        self.run()
    def save_token_Usage(self,mode="read" or "write"):
        def read(self):
            with open(self.token_path, 'r') as f:
                self.token_usage = json.load(f)
        def write(self):
            with open(self.token_path, 'w') as f:
                json.dump(self.token_usage, f, indent=2)
        if mode=="read":
            if os.path.isfile(self.token_path):
                read(self)
        else:
            write(self)

    async def on_message(self, message):
        if message.author == self.user:
            return
        try:
            async with message.channel.typing():
                ChatResult = await self.openai_response(message)
                response = '>>> '
                response += ChatResult["response"]
                response += "\n```"
                response += f"Atokens : {self.token_usage['total_tokens']}\n"
                response += f"IToken : {self.token_usage['prompt_tokens']}\n"
                response += f"CToken : {self.token_usage['completion_tokens']:.02f}\n"
                response += f"TCost : {self.token_usage['total_cost']:.02f}\n"
                response += f"ETa : {ChatResult['time_estimated']}"
                response += "```"
                await message.channel.send(response)
        except Exception as e:
            await message.channel.send(''.join(traceback.TracebackException.from_exception(e).format()))

    async def openai_response(self, message) -> dict["response": str, "time_estimated": str]:
        content = ""+message.content
        if len(message.attachments) != []:
            for i in message.attachments:
                content += f'\nattachment ({i.filename})\n'
                content += str(await discord.message.Attachment.read(i))

        s_time = datetime.now()
        response = ""
        e_time = datetime.now()
        with get_openai_callback() as cb:
            response = agent(content)['output']
            e_time = datetime.now()
            self.token_usage["prompt_tokens"] += cb.prompt_tokens
            self.token_usage["completion_tokens"] += cb.completion_tokens
            self.token_usage["total_tokens"] = self.token_usage["prompt_tokens"] + \
                                               self.token_usage["completion_tokens"]
            self.token_usage["total_cost"] += cb.total_cost
            self.save_token_Usage("write")

        time_estimated = e_time - s_time

        return {"response": response, "time_estimated": period(time_estimated, "{m} m : {s} s")}

    def run(self):
        super().run(self.discord_bot_token)


if __name__ == "__main__":
    os.chdir(os.environ.get('gpt_working_dir'))
    bot = OpenAIBot()
    bot.run()
