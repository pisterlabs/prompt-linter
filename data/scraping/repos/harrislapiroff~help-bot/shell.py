from helpbot.bot.openai import OpenAIReactBot


__all__ = ['ShellBot']


class ShellBot(OpenAIReactBot):
    async def respond(self, message):
        print(message)

    async def send_message(self, message: str, context: dict = {}):
        await super().send_message(message)
        print(message)

    async def run(self):
        while True:
            message = input(">>> ")
            await self.on_message(message)