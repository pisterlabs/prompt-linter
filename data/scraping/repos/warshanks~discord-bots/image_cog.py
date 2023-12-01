"""
This module provides a Discord bot with the capability to generate images using both
OpenAI's Dall-error_message model and Stability.AI's Stable Diffusion model.

The bot uses discord.py's command extension to create commands that users can invoke in chat.

The image generation models are accessible through the ImageCog class, which contains
two commands:
- 'dall-e': Generates an image from a text prompt using OpenAI's Dall-e model.
- 'stable': Generates an image from a text prompt using Stability.AI's Stable Diffusion model.

API keys and other configurations for OpenAI and Stability.AI are loaded from a separate
config module.
"""

import base64
import datetime
import io
import os
from PIL import Image

import openai
import discord
from discord.ext import commands
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from config import openai_token, openai_org, stability_host, stability_token

bot = commands.Bot(command_prefix="~", intents=discord.Intents.all())

# Set OpenAI API key and organization
openai.api_key = openai_token
openai.organization = openai_org

os.environ['STABILITY_HOST'] = stability_host
os.environ['STABILITY_API_KEY'] = stability_token

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=stability_token,  # API Key reference.
    verbose=True,  # Print debug messages.
    engine="stable-diffusion-xl-beta-v2-2-2"
)


# noinspection PyShadowingNames
class ImageCog(commands.Cog):
    """A Cog (a collection of commands) for image generation."""

    def __init__(self, bot):
        """Initialize the ImageCog with a bot instance.

        Args:
            bot (commands.Bot): The bot instance for this cog.
        """
        self.bot = bot

    # Define a slash command that sends a prompt for image generation to OpenAI
    @bot.tree.command(name='dall-e',
                      description="Generate an image from a prompt using "
                                  "OpenAI's Dall-e model")
    async def generate_image_dalle(self, ctx, img_prompt: str):
        """Generate an image using OpenAI's Dall-error_message model.

        Args:
            ctx (commands.Context): The context in which the command was called.
            img_prompt (str): The prompt from which to generate an image.
        """

        # Defer the response to let the user know that the bot is working on the request
        await ctx.response.defer(thinking=True, ephemeral=False)
        # Print information about the user, guild and channel where the command was invoked
        print(ctx.user, ctx.guild, ctx.channel, img_prompt)
        try:
            # Generate an image using OpenAI API from the prompt provided by the user
            response = await openai.Image.acreate(
                prompt=img_prompt,
                size='1024x1024',
                n=1,
                response_format='b64_json'
            )
            base64_img = response['data'][0]["b64_json"]
            img_bytes = base64.b64decode(base64_img)
            decoded_img = Image.open(io.BytesIO(img_bytes))
            decoded_img.save("./images/dalle.png")

            # Send the generated image URL back to the user
            await ctx.followup.send(content=f'"{img_prompt}" '
                                            f'by KC & {ctx.user} '
                                            f'c. {datetime.datetime.now().year}',
                                    file=discord.File(fp="./images/dalle.png"))
        except Exception as error_message:
            await ctx.followup.send(error_message)

    # Define a slash command that sends a prompt for image generation to Stability.AI
    @bot.tree.command(name='stable',
                      description="Generate an image from a prompt using "
                                  "Stability.AI's Stable Diffusion model")
    async def generate_image_stable(self, ctx, img_prompt: str, img_seed: int = None,
                                    img_steps: int = 30, cfg_mod: float = 8.0,
                                    img_width: int = 512, img_height: int = 512):
        """Generate an image using Stability.AI's Stable Diffusion model.

        Args:
            ctx (commands.Context): The context in which the command was called.
            img_prompt (str): The prompt from which to generate an image.
            img_seed (int, optional): The seed for the image generation. Defaults to None.
            img_steps (int, optional): The number of steps for the image generation. Defaults to 30.
            cfg_mod (float, optional): The config scale for the image generation. Defaults to 8.0.
            img_width (int, optional): The width of the generated image. Defaults to 512.
            img_height (int, optional): The height of the generated image. Defaults to 512.
        """

        # Defer the response while the bot processes the image generation request
        await ctx.response.defer(thinking=True, ephemeral=False)

        # Set up the initial image generation parameters using the provided arguments
        answers = stability_api.generate(
            prompt=img_prompt,
            seed=img_seed,
            steps=img_steps,
            cfg_scale=cfg_mod,
            width=img_width,
            height=img_height,
            samples=1,
        )

        # If the adult content classifier is not triggered, send the generated images
        try:
            # Iterate through the answers from the image generation API
            for resp in answers:
                # Iterate through the artifacts in each answer
                for artifact in resp.artifacts:
                    # If the artifact's finish_reason is FILTER,
                    # the API's safety filters have been activated
                    if artifact.finish_reason == generation.FILTER:
                        ctx.followup.send(
                            "Your request activated the API's safety filters "
                            "and could not be processed. "
                            "Please modify the prompt and try again.")
                    # If the artifact's type is an image, send the generated image
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        # Convert the binary artifact into an in-memory binary stream
                        img = io.BytesIO(artifact.binary)
                        img.seek(0)  # Set the stream position to the beginning
                        # Send the generated image as a follow-up message in the chat
                        await ctx.followup.send(content=f'"{img_prompt}" by KC & {ctx.user} '
                                                        f'c.{datetime.datetime.now().year}\n'
                                                        f'Seed: {artifact.seed}',
                                                file=discord.File(fp=img,
                                                                  filename=str(artifact.seed) + ".png"))
        # Catch any exceptions and send an ephemeral message with the error
        except Exception as error_message:
            await ctx.followup.send(error_message)
