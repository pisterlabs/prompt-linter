import os
import openai
import httpx
from discord.ext import commands
from redbot.core import commands
import aiohttp
import discord
import logging
from io import BytesIO
import cv2  # We're using OpenCV to read video
import base64
from collections import defaultdict
import requests
from .constants import CHATTY_INSTRUCTIONS, OPENAI_API_KEY
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.DEBUG)
openai.api_key = OPENAI_API_KEY
conversations = defaultdict(lambda : defaultdict(list))
MAX_HISTORY = 20 
TTS_VOICE = "onyx"
INSTRUCTIONS = "You're a Discord user named 'Cute Recon' in a Discord server to assist other users."
is_busy = False
__author__ = "Teemo the Yiffer"

class YifferGPT(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.use_reactions = True
        self.session = aiohttp.ClientSession(loop=self.bot.loop)
    
    def split_response(self, response, max_length=1900):
        words = response.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(" ".join(current_chunk)) + len(word) + 1 > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def download_image(self, image_url, save_as):
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
        with open(save_as, "wb") as f:
            f.write(response.content)

    async def process_image_link(self, image_url):
        temp_image = "temp_image.jpg"
        await self.download_image(image_url, temp_image)
        output = await self.query(temp_image)
        os.remove(temp_image)
        return output

    def openai_tts(self, input):
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1-hd",
            "input": input,
            "voice": TTS_VOICE
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # Instead of saving the file, we return a BytesIO object so it can run anywhere
            return BytesIO(response.content)
        else:
            # Handle errors
            return f"Error: {response.status_code} - {response.text}"

    def ask_gpt(self, model, message, instruction=INSTRUCTIONS, combined_input=None, history=True):
        # Use a defaultdict where each value defaults to another defaultdict that defaults to an empty list
        global conversations
        if message.author.id not in conversations:
            conversations[message.author.id] = defaultdict(list)

        # Limit the conversation history
        conversations[message.author.id][message.channel.id] = conversations[message.author.id][message.channel.id][-MAX_HISTORY:]
        if combined_input:
            message_text = combined_input
        else:
            message_text = message.content
        if history:
            # Append the user's message to the conversation history
            conversations[message.author.id][message.channel.id].append({"role": "user", "content": message_text})
            history = conversations[message.author.id][message.channel.id]
        else:
            history = [{"role": "user", "content": message_text}]
        prompt = [
                    {
                        "role": "system",
                        "content": instruction
                    },
                    *history
                ]

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=0.8,
                max_tokens=2048, # Max Tokens 4,096
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        except Exception as e:
            return f"Error: {e}"
        
        # Extract the assistant's response
        assistant_message = response['choices'][0]['message']['content'].strip()
        
        # Append the assistant's response to the conversation history
        conversations[message.author.id][message.channel.id].append({"role": "assistant", "content": assistant_message})

        # Re-writing history images to post_filler_image_url
        for message in conversations[message.author.id][message.channel.id]:
            if 'content' in message and isinstance(message['content'], list):
                for content_item in message['content']:
                    if isinstance(content_item, dict) and content_item.get('type') == 'image_url':
                        logging.info("Overwriting URL...")  # For debugging
                        content_item['image_url']['url'] = 'post_filler_image_url'

        return assistant_message

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return
        if self.bot.user.id != message.author.id:
            global is_busy
            if message.channel.id == 1105033083956248576: # regular chatgpt
                if is_busy:
                    print(f"ChatGPT responding to {message.author.id}.")
                    return
                async with message.channel.typing():
                    response = self.ask_gpt("gpt-4-1106-preview", message, CHATTY_INSTRUCTIONS)
                await message.channel.send(response)
            if message.channel.id == 1171379225278820391: # vision
                if is_busy:
                    print(f"ChatGPT responding to {message.author.id}.")
                    return
                async with message.channel.typing():
                    if message.attachments:
                        attachment = message.attachments[0].url
                        ext = message.attachments[0].url.split("/")[-1]
                        #async with self.session.get(attachment) as resp:
                        base64_image = base64.b64encode(requests.get(attachment).content).decode('utf-8')
                        #file = discord.File(BytesIO(data),filename=ext)
                        extra = [
                                    {
                                        "type": "text",
                                        "text": message.content
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                    else:
                        extra = {}
                    response = self.ask_gpt("gpt-4-vision-preview", message, combined_input=extra)
                await message.channel.send(response)
            if message.channel.id == 1171382069482508389: # tts
                if is_busy: 
                    print(f"ChatGPT responding to {message.author.id}.")
                    return
                
                user_query = message.content
                async with message.channel.typing():
                    attachment = message.attachments[0].url
                    ext = message.attachments[0].url.split("/")[-1]
                    async with self.session.get(attachment) as resp:
                        data = await resp.read()
                    #video_file = discord.File(BytesIO(data),filename=ext)
                    video = cv2.VideoCapture(attachment)
                    base64Frames = []
                    while video.isOpened():
                        success, frame = video.read()
                        if not success:
                            break
                        _, buffer = cv2.imencode(".jpg", frame)
                        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

                    video.release()
                    logging.info(len(base64Frames), "frames read.")
                    if not user_query:
                        user_query = "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video."
                    combined_input = [
                                user_query,
                                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
                            ]
                    instruction = "You're a Discord user named 'Cute Recon' who works as a voice actor. While you're great at reading scripts and descriptions of scenes, you excel in creatively telling them."
                    result = self.ask_gpt("gpt-4-vision-preview", message, instruction, combined_input, history=False)
                    audio_data = self.openai_tts(result)
                    if not isinstance(audio_data, str):  # Check if the result is not an error message
                        await message.channel.send(file=discord.File(audio_data, filename="meme" +".mp3"))
                    else:
                        await message.channel.send("Sorry an error occured" + audio_data)

    @commands.command()
    async def voice(self, ctx, new_voice):
        #message = ctx.message.content
        #message = message.replace("!voice ",'')
        global TTS_VOICE
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if new_voice.lower() in valid_voices:
            TTS_VOICE = new_voice.lower()
            return await ctx.channel.send(f"Voice changed to {TTS_VOICE}. 🤖" )
        else:
            return await ctx.channel.send(f"🤖 Please use alloy, echo, fable, onyx, nova, or shimmer. CURRENT VOICE: {TTS_VOICE}  MORE INFO:  <https://platform.openai.com/docs/guides/text-to-speech>"  )
    
    @commands.command()
    async def get_voice(self, ctx):
        return await ctx.channel.send(f"Current TTS Voice: `{TTS_VOICE}`" )

    @commands.command()
    async def instruction(self, ctx):
        new_instructions = ctx.message.content.replace("!instruction ",'')
        INSTRUCTIONS = new_instructions
        return await ctx.channel.send(f"Done! My new instructions are: `{INSTRUCTIONS}`" )

    @commands.command()
    async def get_instruction(self, ctx):
        return await ctx.channel.send(f"Current Instructions: `{INSTRUCTIONS}`" )

    @commands.command()
    async def get_history(self, ctx):
        if ctx.message.author.id not in conversations:
            return await ctx.channel.send(f"Found no conversation history from the AI Bot Channels." )
        else:
            logging.info(conversations)
            return await ctx.channel.send(f"Found your conversation history:\n ```{conversations[ctx.message.author.id]}```" )