
from threading import Thread
from typing import Literal, List
import asyncio
import json
import openai
import os
import platform
import random
import re
import requests

from discord import Embed, Intents, Interaction, Message
from discord.app_commands import describe
from discord.ext.commands import Bot, Context
import discord

from ._constants import *
from ._utils import Config, Engine

class Bot(discord.ext.commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='',
            intents=Intents().all(),
            description='ChatGPT in Discord',
        )

        self._config = Config()

    @property
    def config(self):
        return self._config.load()

    async def on_ready(self):
        await self.wait_until_ready()

        print(f'<Bot: {self.user.name} - {self.user.id}>')
        print(f'<Discord API version: {discord.__version__}>')
        print(f'<OpenAI API version: {openai.version.VERSION}>')
        print(f'<Python version: {platform.python_version()}>')
        print(f'<Operating System: {platform.system()} {platform.release()} ({os.name})>')

        # Set up slash commands
        print('[+] Setting up slash commands...')
        await self.tree.sync()
        print('[+] Slash commands are ready now.')

        print('\n--Bot is ready now--\n')

    async def on_message(self, _):
        return

def _main():
    # Initialize discord
    global bot
    bot = Bot()
    asyncio.run(bot.setup_hook())

    # Initialize OpenAI
    openai.api_key = bot.config['openai.key']
    openai.log = 'info'
    openai_config: dict = bot.config['openai.config']

    def load_engines() -> tuple:
        """Load the engines from the config.json file.
        If the config.json file didn't specified, it will load all the available engines from OpenAI API.

        Returns:
            tuple: Tuple of engine IDs
        """
        config_engines_list = openai_config['select_only_these_engines']
        if len(config_engines_list) != 0:
            return tuple(config_engines_list)

        engine = Engine()
        engine_id: list = []
        engines: dict = engine.load()
        if len(engines['data']) > 25:
            print(f'[!] Warning: There are more than 25 engines available. Only the first 25 will be used. It is highly recommended to select only the engines you want to use in the config.json file.')

        for engine in engines['data']:
            if len(engine_id) == 25:
                break

            engine_id.append(engine['id'])

        return tuple(engine_id)

    @bot.tree.command(
        name='chat',
        description='Chat with the AI.'
    )
    @describe(
        engine = f'Default: {openai_config["engine"]}. The engine to use for the chat.',
        ephemeral = 'Whether the response should be ephemeral or not.',
        frequency_penalty = f'Default: {openai_config["frequency_penalty"]}. Number between -2.0 and 2.0. Positive values penalize new tokens.',
        max_tokens = f'Default: {openai_config["max_tokens"]}. The maximum number of tokens to generate in the completion.',
        prompt = 'Anything you want to say to the AI. Have respect to the AI.',
        presence_penalty = f'Default: {openai_config["presence_penalty"]}. Number between -2.0 and 2.0. Positive values penalize new tokens.',
        temperature = f'Default: {openai_config["temperature"]}.Try 0.9 for more creative applications.',
        top_p = f'Default: {openai_config["top_p"]}. An alternative to sampling with temperature, called nucleus sampling.',
        verbose = 'Whether to show the configuration used for the chat.',
    )
    async def chat(
        interaction: Interaction, *, prompt: str,
        ephemeral: bool = False, verbose: bool = False,
        engine: Literal[load_engines()] = MISSING, frequency_penalty: float = MISSING,
        max_tokens: int = MISSING, presence_penalty: float = MISSING,
        temperature: float = MISSING, top_p: float = MISSING,
    ):
        engine = openai_config['engine'] if engine is MISSING else engine
        frequency_penalty = openai_config['frequency_penalty'] if frequency_penalty is MISSING else frequency_penalty
        max_tokens = openai_config['max_tokens'] if max_tokens is MISSING else max_tokens
        presence_penalty = openai_config['presence_penalty'] if presence_penalty is MISSING else presence_penalty
        temperature = openai_config['temperature'] if temperature is MISSING else temperature
        top_p = openai_config['top_p'] if top_p is MISSING else top_p

        embed = Embed(
            title = prompt,
        )
        embed.set_author(
            name = interaction.user.name,
            icon_url = interaction.user.avatar.url
        )
        embed.set_footer(
            text = f'Powered by {engine}',
            icon_url = 'https://seeklogo.com/images/O/open-ai-logo-8B9BFEDC26-seeklogo.com.png'
        )

        if verbose:
            embed.add_field(name='max_tokens', value=max_tokens)
            embed.add_field(name='temperature', value=temperature)
            embed.add_field(name='top_p', value=top_p)
            embed.add_field(name='frequency_penalty', value=frequency_penalty)
            embed.add_field(name='presence_penalty', value=presence_penalty)

        # Response to the interaction by sending the prompt
        await interaction.response.send_message(ephemeral=ephemeral, embed=embed)

        # Create a message to edit later
        message: Message = await interaction.followup.send(
            ephemeral=ephemeral,
            embed=Embed(description='I\'m thinking...')
        )

        try:
            # Create a generator that will yield the response
            ai = openai.Completion.create(
                echo=False,
                stream=True,
                engine=engine,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
            )
        except openai.error.OpenAIError as e:
            # In case of an error, edit the message to show the error
            return await message.edit(
                embed = Embed(
                    title = type(e).__name__,
                    description = e.user_message
                )
            )

        results: str = ''
        def ai_genr():
            """Run the generator and store the results in a variable.
            """
            nonlocal ai, results
            for response in ai:
                text = response['choices'][0].get('text', 'No answer')
                results += text

        # Run the generator in a separate thread
        t_ai_genr = Thread(target=ai_genr)
        t_ai_genr.start()

        def embed_genr() -> List[Embed]:
            """Convert the results into a list of embeds.
            """
            def validURL(string: str) -> str:
                """Return first valid URL in a string.
                """
                regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
                url = re.findall(regex, string)
                urls = [x[0] for x in url]
                if len(urls) == 0:
                    return None

                # Randomly choose one URL from the list
                k = random.choice(urls)
                try:
                    # We only go through one URL at a time to prevent wasting time on testing all URLs
                    requests.get(k)
                    if requests.get(k).status_code == 200:
                        return k
                except:
                    # Might get alot of errors such as MaxRetryError, ConnectionError, SSLCertVerificationError.
                    pass

                return None

            nonlocal results
            embeds = []

            # If there are no results, return an embed with a message
            if len(results.strip()) == 0:
                return [Embed(description='I\'m thinking...')]

            # Since one embed only supports 4096 characters, we need to split the results into multiple embeds
            pages = len(results) // 4096
            for page in range(pages + 1):
                embed = Embed(
                    description = results[page * 4096: (page + 1) * 4096]
                )

                # If the embed has an image link, set the image
                url = validURL(embed.description)
                if url is not None:
                    embed.set_image(url=url)

                embeds.append(embed)

            return embeds

        # Initialize the embed variable
        embeds: list = embed_genr()

        async def send():
            """Send the embeds.
            """
            nonlocal message, embeds

            # Comparing if new embeds are generated
            prev_embeds_len = len(embeds)
            embeds = embed_genr()

            # If new embeds are generated
            if prev_embeds_len != len(embeds):
                # Since one message only can handle around 6000 characters, so we need to send the embed in a new message
                message = await interaction.followup.send(
                    ephemeral=ephemeral,
                    embed=embeds[-1]
                )

            # If no new embeds are generated
            else:
                await message.edit(embed=embeds[-1])

        await asyncio.sleep(1)
        while t_ai_genr.is_alive():
            await send()
        else:
            # If the generator is done, finally resend the embed one last time to ensure nothing is left
            await asyncio.sleep(1)
            await send()

    bot.run(
        bot.config['discord.token'],
        reconnect=True,
        log_level=0,
    )

def main():
    def discordTokenValidator(token: str) -> bool:
        """Check if the discord token is valid.
        """
        response = requests.get(
            'https://discord.com/api/v10/users/@me',
            headers = {
                "Authorization": f'Bot {token}'
            }
        )
        if response.status_code == 200:
            return True
        else:
            return False

    def getOpenAIModels(key: str) -> bool:
        """Check if the OpenAI API key is valid.
        At the same time, get a list of available engines from the API, and write it to the engine file.
        """
        response = requests.get(
            'https://api.openai.com/v1/models',
            headers = {
                "Authorization": f'Bearer {key}'
            }
        )
        if response.status_code == 200:
            engines = response.json()
            engine.write(engines)
            return True
        else:
            return False

    config = Config()
    engine = Engine()
    print(repr(config))
    print(repr(engine))

    # Prompt the user to enter a valid discord token and OpenAI API key
    while True:
        data = config.load()
        if not discordTokenValidator(data['discord.token']):
            data['discord.token'] = input('Discord Login Failure!\nPlease enter a valid discord token: ')

        elif not getOpenAIModels(data['openai.key']):
            data['openai.key'] = input('OpenAI Login Failure!\nPlease enter a valid OpenAI API key: ')

        else:
            break

        config.write(data)

    _main()
