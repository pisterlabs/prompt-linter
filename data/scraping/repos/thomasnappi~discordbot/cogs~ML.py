"ml.py" #pylint: disable=invalid-name
import io
import json
import traceback
import pytesseract
from discord.ext import commands
from PIL import Image
import numpy as np
import openai
from extras.common_lib import send_split, log

def calculate_confidences(conf, words):
    """Determine the proper output given words and confidences."""
    unreadable = False
    last_nl = True
    to_print = ""
    for i,confidence in enumerate(conf):
        if float(confidence) > 50:
            if unreadable:
                to_print += "[unreadable] "
            last_nl = False
            unreadable = False
            to_print += words[i] + " "
        elif float(confidence) > 10:
            if unreadable:
                to_print += "[unreadable] "
            last_nl = False
            unreadable = False
            to_print += "(guess: "+words[i]+"?)" + " "
        elif float(confidence) < 0:
            if last_nl:
                continue
            last_nl = True
            if unreadable:
                to_print += "[unreadable] "
            unreadable = False
            to_print += '\n'
        else:
            unreadable = True
    return to_print

class ML(commands.Cog):
    """Machine learning commands."""
    def __init__(self, client):
        self.client = client
        with open("openai.txt", encoding="utf8") as file:
            openai.api_key = file.readlines()[0].strip()

    @commands.command(name="ocr",pass_context=True)
    async def ocr(self, ctx):
        """ Run OCR on the last image sent in this channel (looks back 20 messages)"""
        async for message in ctx.history(limit=20):
            if len(message.attachments) > 0 and (
                    message.attachments[0].filename.lower()[-4:] in [".png",".jpg",".gif"] or
                    message.attachments[0].filename.lower()[-5:] == ".jpeg"):
                bio = io.BytesIO()
                await message.attachments[0].save(bio)
                img = np.array(Image.open(bio))
                data = pytesseract.image_to_data(img, output_type="dict")
                to_print = calculate_confidences(data["conf"],data["text"])
                await send_split(ctx, to_print)
                return

    @commands.command(name="chatgpt", aliases=["cgpt"], pass_context=True)
    async def chatgpt(self, ctx, *, prompt):
        """
        Generate a response to your prompt with OpenAI's ChatGPT.  Sends data to the OpenAI API.
        Please abide by the terms of use: https://openai.com/terms/
        """
        try:
            await ctx.message.add_reaction("⏰")
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                n=1,
                temperature=0.7,
                stop=None
            )
            message = completion['choices'][0]['message']['content']
            log(json.dumps({
                "user": ctx.author.id,
                "prompt": prompt,
                "received": message
            }))
            try:
                await send_split(ctx.message, message, code=True, reply=True)
            except Exception: #pylint: disable=broad-except
                await send_split(ctx, message, code=True)
        except openai.error.OpenAIError as e:
            await ctx.send(f"Sorry, there was an issue with openai.\n{e}")
            print(e)
        except Exception as e: #pylint: disable=broad-except
            trace : str = traceback.format_exc()
            print(trace)
            print(e)
            try:
                await ctx.send(trace)
            finally:
                await ctx.send(e)

async def setup(client):
    """Setup the cog."""
    await client.add_cog(ML(client))
