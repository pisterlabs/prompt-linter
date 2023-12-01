import asyncio
import os
import random
from dataclasses import dataclass
from typing import Optional

import discord
import requests
# from aleph_alpha_client import AlephAlphaClient
# from aleph_alpha_client import AlephAlphaModel
# from aleph_alpha_client import CompletionRequest
# from aleph_alpha_client import ImagePrompt
# from aleph_alpha_client import Prompt
from discord.ext import commands
from marsbots.discord_utils import is_mentioned
from marsbots.discord_utils import replace_bot_mention
from marsbots.discord_utils import replace_mentions_with_usernames
from marsbots.language_models import OpenAIGPT3LanguageModel
from marsbots_eden.eden import get_file_update
from marsbots_eden.eden import poll_creation_queue
from marsbots_eden.eden import request_creation
from marsbots_eden.models import SignInCredentials
from marsbots_eden.models import SourceSettings
from marsbots_eden.models import StableDiffusionConfig

from . import config
from . import settings

EDEN_API_URL = "https://api.eden.art" # os.getenv("EDEN_API_URL")
EDEN_API_KEY = os.getenv("EDEN_API_KEY")
EDEN_API_SECRET = os.getenv("EDEN_API_SECRET")

CONFIG = config.config_dict[config.stage]
ALLOWED_GUILDS = CONFIG["guilds"]
ALLOWED_CHANNELS = CONFIG["allowed_channels"]
ALLOWED_LERP_BACKDOOR_USERS = CONFIG["allowed_channels"]

# experimental
os.environ['OPENAI_API_KEY'] = os.environ['LM_OPENAI_API_KEY']
from logos.scenarios import EdenAssistant


@dataclass
class GenerationLoopInput:
    api_url: str
    start_bot_message: str
    source: SourceSettings
    config: any
    message: discord.Message
    is_video_request: bool = False
    prefer_gif: bool = True
    refresh_interval: int = 2
    parent_message: discord.Message = None


class LerpModal(discord.ui.Modal):
    def __init__(self, bot, refresh_callback, loop_input, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bot = bot
        self.refresh_callback = refresh_callback
        self.loop_input = loop_input
        self.add_item(discord.ui.InputText(label="Short Input"))

    async def callback(self, interaction: discord.Interaction):
        ctx = await self.bot.get_application_context(interaction)
        await ctx.defer()
        text_input1 = self.loop_input.config.text_input
        text_input2 = self.children[0].value
        interpolation_texts = [text_input1, text_input2]
        seed1 = self.loop_input.config.seed
        seed2 = random.randint(1, 1e8)
        interpolation_seeds = [seed1, seed2]
        width = self.loop_input.config.width
        height = self.loop_input.config.height
        n_frames = 50
        steps = self.loop_input.config.steps
        self.loop_input.config = StableDiffusionConfig(
            generator_name="interpolate",
            stream=True,
            stream_every=1,
            text_input=[f'{text_input1} to {text_input2}'],
            interpolation_texts=interpolation_texts,
            interpolation_seeds=interpolation_seeds,
            n_frames=n_frames,
            width=width,
            height=height,
            steps=steps,
        )
        self.loop_input.is_video_request = True
        await self.refresh_callback(loop_input=self.loop_input, reroll_seed=False)


class CreationActionButtons(discord.ui.View):
    def __init__(
        self,
        *items,
        bot,
        output_url,
        refresh_callback,
        loop_input: GenerationLoopInput,
        timeout=180,
    ):
        super().__init__(*items, timeout=timeout)
        self.bot = bot
        self.output_url = output_url
        self.refresh_callback = refresh_callback
        self.loop_input = loop_input

    # this needs to be adapted to api reactions
    async def feedback(self, stat, interaction):
        ctx = await self.bot.get_application_context(interaction)
        await ctx.defer()
        requests.post(
            self.loop_input.api_url + "/update_stats",
            json={
                "creation": self.output_url,
                "stat": stat,
                "opperation": "increase",
                "address": interaction.user.id,
            },
        )

    @discord.ui.button(emoji="🔄", style=discord.ButtonStyle.blurple)
    async def refresh(self, button, interaction):
        ctx = await self.bot.get_application_context(interaction)
        await ctx.defer()
        await self.refresh_callback(
            loop_input=self.loop_input,
        )

    @discord.ui.button(label="Lerp It")
    async def lerp(self, button, interaction):
        await interaction.response.send_modal(
            LerpModal(
                title="Lerp It",
                bot=self.bot,
                refresh_callback=self.refresh_callback,
                loop_input=self.loop_input,
            )
        )

    @discord.ui.button(emoji="🔥", style=discord.ButtonStyle.red)
    async def burn(self, button, interaction):
        await self.feedback("burn", interaction)

    @discord.ui.button(label="🙌", style=discord.ButtonStyle.green)
    async def praise(self, button, interaction):
        await self.feedback("praise", interaction)
        self.stop()


class EdenCog(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.eden_credentials = SignInCredentials(
            apiKey=EDEN_API_KEY, apiSecret=EDEN_API_SECRET
        )
        self.language_model = OpenAIGPT3LanguageModel(
            engine=settings.GPT3_ENGINE,
            temperature=settings.GPT3_TEMPERATURE,
            frequency_penalty=settings.GPT3_FREQUENCY_PENALTY,
            presence_penalty=settings.GPT3_PRESENCE_PENALTY,
        )
        self.assistant = EdenAssistant("gpt-4")

    @commands.slash_command(guild_ids=ALLOWED_GUILDS)
    async def create(
        self,
        ctx,
        text_input: discord.Option(str, description="Prompt", required=True),
        aspect_ratio: discord.Option(
            str,
            choices=[
                discord.OptionChoice(name="square", value="square"),
                discord.OptionChoice(name="landscape", value="landscape"),
                discord.OptionChoice(name="portrait", value="portrait"),
            ],
            required=False,
            default="square",
        ),
        # large: discord.Option(
        #     bool,
        #     description="Larger resolution, ~2.25x more pixels",
        #     required=False,
        #     default=False,
        # ),
        # fast: discord.Option(
        #     bool,
        #     description="Fast generation, possibly some loss of quality",
        #     required=False,
        #     default=False,
        # ),
    ):
        print("Received create:", text_input)

        if not self.perm_check(ctx):
            await ctx.respond("This command is not available in this channel.")
            return

        if settings.CONTENT_FILTER_ON:
            if not OpenAIGPT3LanguageModel.content_safe(text_input):
                await ctx.respond(
                    f"Content filter triggered, <@!{ctx.author.id}>. Please don't make me draw that. If you think it was a mistake, modify your prompt slightly and try again.",
                )
                return

        source = self.get_source(ctx)
        large, fast = False, False
        width, height, upscale_f = self.get_dimensions(aspect_ratio, large)
        steps = 15 if fast else 40

        config = StableDiffusionConfig(
            generator_name="create",
            text_input=text_input,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=7.5,
            upscale_f=upscale_f,
            seed=random.randint(1, 1e8),
        )

        start_bot_message = f"**{text_input}** - <@!{ctx.author.id}>\n"
        await ctx.respond("Starting to create...")
        message = await ctx.channel.send(start_bot_message)

        generation_loop_input = GenerationLoopInput(
            api_url=EDEN_API_URL,
            message=message,
            start_bot_message=start_bot_message,
            source=source,
            config=config,
            is_video_request=False
        )
        await self.generation_loop(generation_loop_input)

    @commands.slash_command(guild_ids=ALLOWED_GUILDS)
    async def remix(
        self,
        ctx,
        image1: discord.Option(
            discord.Attachment, description="Image to remix", required=True
        ),
    ):
        print("Received remix:", image1)

        if not self.perm_check(ctx):
            await ctx.respond("This command is not available in this channel.")
            return

        if not image1:
            await ctx.respond("Please provide an image to remix.")
            return

        source = self.get_source(ctx)

        steps = 50
        width, height = 1024, 1024

        config = StableDiffusionConfig(
            generator_name="remix",
            text_input="remix",
            init_image_data=image1.url,
            init_image_strength=0.125,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=7.5,
            seed=random.randint(1, 1e8)
        )

        start_bot_message = f"**Remix** by <@!{ctx.author.id}>\n"
        await ctx.respond("Remixing...")
        message = await ctx.channel.send(start_bot_message)

        generation_loop_input = GenerationLoopInput(
            api_url=EDEN_API_URL,
            message=message,
            start_bot_message=start_bot_message,
            source=source,
            config=config,
            is_video_request=False,
            prefer_gif=False
        )
        await self.generation_loop(generation_loop_input)

    @commands.slash_command(guild_ids=ALLOWED_GUILDS)
    async def real2real(
        self,
        ctx,
        image1: discord.Option(
            discord.Attachment, description="First image", required=True
        ),
        image2: discord.Option(
            discord.Attachment, description="Second image", required=True
        ),
    ):

        if not self.perm_check(ctx):
            await ctx.respond("This command is not available in this channel.")
            return

        source = self.get_source(ctx)

        if not (image1 and image2):
            await ctx.respond("Please provide two images to interpolate between.")
            return

        interpolation_init_images = [image1.url, image2.url]

        interpolation_seeds = [
            random.randint(1, 1e8) for _ in interpolation_init_images
        ]
        n_frames = 50
        steps = 40
        width, height = 768, 768

        config = StableDiffusionConfig(
            generator_name="real2real",
            stream=True,
            stream_every=1,
            text_input="real2real",
            interpolation_seeds=interpolation_seeds,
            interpolation_init_images=interpolation_init_images,
            interpolation_init_images_use_img2txt=True,
            n_frames=n_frames,
            loop=True,
            smooth=True,
            n_film=1,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=6.5,
            seed=random.randint(1, 1e8),
            interpolation_init_images_min_strength = 0.3,  # a higher value will make the video smoother, but allows less visual change / journey
        )

        start_bot_message = f"**Real2Real** by <@!{ctx.author.id}>\n"
        await ctx.respond("Lerping...")
        message = await ctx.channel.send(start_bot_message)

        generation_loop_input = GenerationLoopInput(
            api_url=EDEN_API_URL,
            message=message,
            start_bot_message=start_bot_message,
            source=source,
            config=config,
            is_video_request=True,
            prefer_gif=False,
        )
        await self.generation_loop(generation_loop_input)

    @commands.slash_command(guild_ids=ALLOWED_GUILDS)
    async def lerp(
        self,
        ctx,
        text_input1: discord.Option(str, description="First prompt", required=True),
        text_input2: discord.Option(str, description="Second prompt", required=True),
        aspect_ratio: discord.Option(
            str,
            choices=[
                discord.OptionChoice(name="square", value="square"),
                discord.OptionChoice(name="landscape", value="landscape"),
                discord.OptionChoice(name="portrait", value="portrait"),
            ],
            required=False,
            default="square",
        ),
    ):
        print("Received lerp:", text_input1, text_input2)

        if not self.perm_check(ctx):
            await ctx.respond("This command is not available in this channel.")
            return

        if settings.CONTENT_FILTER_ON:
            if not OpenAIGPT3LanguageModel.content_safe(
                text_input1,
            ) or not OpenAIGPT3LanguageModel.content_safe(text_input2):
                await ctx.respond(
                    f"Content filter triggered, <@!{ctx.author.id}>. Please don't make me draw that. If you think it was a mistake, modify your prompt slightly and try again.",
                )
                return

        source = self.get_source(ctx)

        interpolation_texts = [text_input1, text_input2]
        interpolation_seeds = [random.randint(1, 1e8) for _ in interpolation_texts]
        n_frames = 50
        steps = 40
        width, height, upscale_f = self.get_video_dimensions(aspect_ratio, False)

        config = StableDiffusionConfig(
            generator_name="interpolate",
            stream=True,
            stream_every=1,
            text_input=text_input1,
            interpolation_texts=interpolation_texts,
            interpolation_seeds=interpolation_seeds,
            n_frames=n_frames,
            smooth=True,
            loop=True,
            n_film=1,
            width=width,
            height=height,
            sampler="euler",
            steps=steps,
            guidance_scale=7.5,
            seed=random.randint(1, 1e8),
        )

        start_bot_message = (
            f"**{text_input1}** to **{text_input2}** - <@!{ctx.author.id}>\n"
        )
        await ctx.respond("Lerping...")
        message = await ctx.channel.send(start_bot_message)

        generation_loop_input = GenerationLoopInput(
            api_url=EDEN_API_URL,
            message=message,
            start_bot_message=start_bot_message,
            source=source,
            config=config,
            is_video_request=True,
            prefer_gif=False,
        )
        await self.generation_loop(generation_loop_input)

    async def generation_loop(
        self,
        loop_input: GenerationLoopInput,
    ):
        api_url = loop_input.api_url
        start_bot_message = loop_input.start_bot_message
        parent_message = loop_input.parent_message
        message = loop_input.message
        source = loop_input.source
        config = loop_input.config
        refresh_interval = loop_input.refresh_interval
        is_video_request = loop_input.is_video_request
        prefer_gif = loop_input.prefer_gif

        try:
            task_id = await request_creation(
                api_url, self.eden_credentials, source, config
            )
            current_output_url = None
            while True:
                result, file, output_url = await poll_creation_queue(
                    api_url, self.eden_credentials, task_id, is_video_request, prefer_gif
                )
                if output_url != current_output_url:
                    current_output_url = output_url
                    message_update = self.get_message_update(result)
                    await self.edit_message(
                        message,
                        start_bot_message,
                        message_update,
                        file_update=file,
                    )
                if result["status"] == "completed":
                    file, output_url = await get_file_update(
                        result, is_video_request, prefer_gif
                    )
                    view = CreationActionButtons(
                        bot=self.bot,
                        output_url=output_url,
                        loop_input=loop_input,
                        refresh_callback=self.refresh_callback,
                    )
                    if parent_message:
                        new_message = await parent_message.reply(
                            start_bot_message,
                            files=[file],
                            view=None,
                        )
                    else:
                        new_message = await message.channel.send(
                            start_bot_message,
                            files=[file],
                            view=None,
                        )
                    view.loop_input.parent_message = new_message
                    await message.delete()
                    return
                await asyncio.sleep(refresh_interval)

        except Exception as e:
            await self.edit_message(message, start_bot_message, f"Error: {e}")

    async def refresh_callback(
        self,
        loop_input: GenerationLoopInput,
        reroll_seed: bool = True,
    ):
        loop_input.message = await loop_input.parent_message.reply(
            loop_input.start_bot_message,
        )
        if reroll_seed:
            loop_input.config.seed = random.randint(1, 1e8)
        await self.generation_loop(loop_input)

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        try:
            if (
                message.channel.id not in ALLOWED_CHANNELS
                or message.author.id == self.bot.user.id
                or message.author.bot
            ):
                return

            trigger_reply = is_mentioned(message, self.bot.user)

            if trigger_reply:
                ctx = await self.bot.get_context(message)
                async with ctx.channel.typing():
                    prompt = self.message_preprocessor(message)
                    
                    attachment_urls = [attachment.url for attachment in message.attachments]
                    attachment_lookup_file = {url: f"/files/image{i+1}.jpeg" for i, url in enumerate(attachment_urls)}
                    attachment_lookup_url = {v: k for k, v in attachment_lookup_file.items()}
                    attachment_files = [attachment_lookup_file[url] for url in attachment_urls]
      
                    assistant_message = {
                        "prompt": prompt,
                        "attachments" : attachment_files
                    }

                    response = await self.assistant(
                        assistant_message, 
                        session_id=str(message.author.id)
                    )
                    
                    reply = response["message"][:2000]
                    reply_message = await message.reply(reply)

                    # check if there is a config
                    config = response["attachment"]
                    if not config:
                        return

                    mode = config.pop("generator")

                    if 'text_input' in config:
                        text_input = config['text_input']
                    elif 'interpolation_texts' in config:
                        text_input = " to ".join(config['interpolation_texts'])
                    else:
                        text_input = mode

                    if 'init_image_data' in config:
                        config['init_image_data'] = attachment_lookup_url[config['init_image_data']]
                    if 'interpolation_init_images' in config:
                        config['interpolation_init_images'] = [
                            attachment_lookup_url[img] for img in config['interpolation_init_images']
                        ]

                    config = StableDiffusionConfig(
                        generator_name=mode,
                        seed=random.randint(1, 1e8),
                        **config
                    )
                    
                    source = self.get_source(ctx)

                    is_video_request = mode in ["interpolate", "real2real"]
                    
                    start_bot_message = f"**{text_input}** - <@!{ctx.author.id}>\n"
                    original_text = f"{reply[0:1950-len(start_bot_message)]}\n\n{start_bot_message}"
                    
                    generation_loop_input = GenerationLoopInput(
                        api_url=EDEN_API_URL,
                        message=reply_message,
                        start_bot_message=original_text,
                        source=source,
                        config=config,
                        prefer_gif=False,
                        is_video_request=is_video_request
                    )
                    await self.generation_loop(generation_loop_input)

        except Exception as e:
            print(f"Error: {e}")
            await message.reply(":) ")

    def message_preprocessor(self, message: discord.Message) -> str:
        message_content = replace_bot_mention(message.content, only_first=True)
        message_content = replace_mentions_with_usernames(
            message_content,
            message.mentions,
        )
        message_content = message_content.strip()
        return message_content

    def get_video_dimensions(self, aspect_ratio):
        if aspect_ratio == "square":
            width, height = 1024, 1024
        elif aspect_ratio == "landscape":
            width, height = 1280, 768
        elif aspect_ratio == "portrait":
            width, height = 768, 1280
        upscale_f = 1.0
        return width, height, upscale_f

    def get_dimensions(self, aspect_ratio, large):
        if aspect_ratio == "square":
            width, height = 768, 768
        elif aspect_ratio == "landscape":
            width, height = 960, 640
        elif aspect_ratio == "portrait":
            width, height = 640, 960
        upscale_f = 1.4 if large else 1.0
        return width, height, upscale_f

    def perm_check(self, ctx):
        if ctx.channel.id not in ALLOWED_CHANNELS:
            return False
        return True

    def get_source(self, ctx):
        source = SourceSettings(
            author_id=int(ctx.author.id),
            author_name=str(ctx.author),
            guild_id=int(ctx.guild.id),
            guild_name=str(ctx.guild),
            channel_id=int(ctx.channel.id),
            channel_name=str(ctx.channel),
        )
        return source

    def get_message_update(self, result):
        status = result["status"]
        if status == "failed":
            return "_Server error: Eden task failed_"
        elif status in "pending":
            return "_Warming up, please wait._"
        elif status in "starting":
            return "_Creation is starting_"
        elif status == "running":
            progress = int(100 * result["progress"])
            return f"_Creation is **{progress}%** complete_"
        elif status == "complete":
            return "_Creation is **100%** complete_"

    async def edit_interaction(
        self,
        ctx,
        start_bot_message,
        message_update,
        file_update=None,
    ):
        message_content = f"{start_bot_message}\n{message_update}"
        if file_update:
            await ctx.edit(content=message_content, file=file_update)
        else:
            await ctx.edit(content=message_content)

    async def edit_message(
        self,
        message: discord.Message,
        start_bot_message: str,
        message_update: str,
        file_update: Optional[discord.File] = None,
    ) -> discord.Message:
        if message_update is not None:
            message_content = f"{start_bot_message}\n{message_update}"
            await message.edit(content=message_content)            
        if file_update:
            await message.edit(files=[file_update], attachments=[])


def setup(bot: commands.Bot) -> None:
    bot.add_cog(EdenCog(bot))
