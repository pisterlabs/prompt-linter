import discord
from discord.ext import commands as cmds
from dotenv import load_dotenv
from typing import Optional, List, Literal
import requests
import asyncio
import openai
import json
import time
import os
import config

load_dotenv()

# settings

directory_path = './prompts'
characters = [name[:-4] for name in os.listdir(directory_path)]
timescoped = 5

# variables

openai_key = os.getenv('OPENAI_KEY')
openai.api_key = openai_key

# async functions

class Tchat(cmds.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.rdy = 0
        self.payload = None
        self.reply_lang = 'th'
        self.char_is_set = False
        self.perm = None
        self.messages = None
        self.save_foldername = None
        self.char = None
        self.dialogue = 'prompts'
        self.memIdx = None
        self.voice = None
        self.user_id = None
        self.suffix = None
        self.vc_playing = False
        self.SPAM_COOLDOWN = 6
        self.last_message_time = None

    def getSuffix(self):
        os.makedirs(self.save_foldername, exist_ok=True)
        base_filename = 'conversation'
        count = 0
        filename = os.path.join(self.save_foldername, f'{base_filename}_{count}.txt')
        while os.path.exists(filename):
            count += 1
            filename = os.path.join(self.save_foldername, f'{base_filename}_{count}.txt')
        with open(filename, 'w', encoding = 'utf-8') as file:
            json.dump(self.messages, file, indent=4, ensure_ascii=False)
        return count

    def trans(self, say):
        url = "https://translated-mymemory---translation-memory.p.rapidapi.com/get"
        querystring = {"langpair":"en|th","q":say,"mt":"1","onlyprivate":"0","de":"a@b.c"}
        headers = {
            "X-RapidAPI-Key": "1f678aafcdmshd6d45bc81882a00p18750djsnefc88f21f555",
            "X-RapidAPI-Host": "translated-mymemory---translation-memory.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        data = json.loads(response.text).get('responseData').get('translatedText')
        return data       

    def voiceGen(self, data): # wip
        url = "https://freetts.com/api/TTS/SynthesizeText"
        payload = json.dumps({
        "text": data,
        "type": 0,
        "ssml": 0,
        "isLoginUser": 0,
        "country": "Japanese (Japan)",
        "voiceType": "Standard",
        "languageCode": "ja-JP",
        "voiceName": "ja-JP-Standard-A",
        "gender": "FEMALE"
        })
        headers = {
        'authority': 'freetts.com',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en,th-TH;q=0.9,th;q=0.8',
        'authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1NDUyNDIwOTksImlhdCI6MTY4MTI0MjA4OSwiaXNzIjoia2VuIiwiZGF0YSI6eyJ1c2VybmFtZSI6IjE3Mi42OC4yNDIuMjQ4IiwiaWQiOiIxNzIuNjguMjQyLjI0OCIsImxvZ2luX3RpbWUiOjE2ODEyNDIwODl9fQ.btWVQw4ygWKTBjJ9nBhX2txZ6jlipIB49EYDNeEXZmU',
        'content-type': 'application/json',
        'cookie': '_ga=GA1.2.1262120938.1679757155; _gid=GA1.2.1415340467.1681226232; _gat=1',
        'origin': 'https://freetts.com',
        'referer': 'https://freetts.com/',
        'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        res = json.loads(response.text)['data']['audiourl']
        return res
        
    def textGen(self):
        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": self.messages,
        "max_tokens": 50
        })
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_key}'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text).get('choices')[0].get('message').get('content')
        self.messages.append({"role": "assistant", "content": f"{data}"})
        return data

    def memorize(self):
        os.makedirs(self.save_foldername, exist_ok=True)
        base_filename = 'conversation'
        filename = os.path.join(self.save_foldername, f'{base_filename}_{self.suffix}.txt')
        with open(filename, 'w', encoding = 'utf-8') as file:
            json.dump(self.messages, file, indent=4, ensure_ascii=False)

    @cmds.command()
    async def stop(self, ctx):
        if self.vc_playing == False:
            self.rdy = 0
            print("\nleaving:")
            await ctx.channel.send("> leaving..")
            await self.payload.send("``` voice chat off ```")
            await self.payload.voice_client.disconnect(force=True)
            try:
                self.messages.append({"role" : "assistant", "content" : "yes"})
                self.memorize()
            except:
                print("memorize failed")
                pass

    @cmds.Cog.listener()
    async def on_message(self, ctx):
        if ctx.author.bot:
                return
        
        if self.rdy:

            if self.last_message_time is not None:
                time_diff = (ctx.created_at - self.last_message_time).total_seconds()

                if time_diff < self.SPAM_COOLDOWN:
                    self.vc_playing = False
                    self.last_message_time = ctx.created_at
                    return
                
            self.last_message_time = ctx.created_at
            await  self.bot.process_commands(ctx)
            self.vc_playing = True
            
            start_time = time.time()
            text = ctx.content

            await ctx.channel.send("> listening")
            print("\nlistening:")

            try:
                channel_id = ctx.channel.id
                if ctx.channel.id == channel_id:
                    self.messages.append({"role" : "user", "content" : text})

                    try:
                        await ctx.channel.send("> generating")
                        print("\ngenerating:")
                        say = self.textGen()
                        await ctx.channel.send(say)

                        if self.reply_lang == 'th':
                            audio_stream = f'https://tipme.in.th/api/tts/?text={self.trans(say)}&format=opus'
                        elif self.reply_lang == 'en':
                            audio_stream = f'https://api.streamelements.com/kappa/v2/speech?voice=Brian&text={say}'

                        source = discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(audio_stream))
                        self.payload.voice_client.play(source)

                        while self.payload.voice_client.is_playing():
                            await asyncio.sleep(1)
                            
                        start_time = time.time()
                    except Exception as e:
                        print(f"{e}")
                        print("Token limit exceeded")

            except:
                if time.time() - start_time > timescoped:
                    await ctx.channel.send("> ~listening")
                    print("\n~listening:")

        self.vc_playing = False

                
    
    @cmds.hybrid_command(description=f'characters: {characters}')
    async def set_tchat(self, ctx, char: str, dialogue: Optional[Literal['new', 'con']] = 'prompts'):
        if ctx.author.id not in config.OWNER:
            return await ctx.reply("> you have no perm to use this command!")
        
        self.save_foldername = f'history/{char}'
        self.char = char
        
        if dialogue == 'con':
            self.dialogue = 'history'
            try:
                count = 0
                filename = os.path.join(self.save_foldername, f'conversation_{count}.txt')
                while os.path.exists(filename):
                    count += 1
                    filename = os.path.join(self.save_foldername, f'conversation_{count}.txt')
                if count == 0:
                    raise Exception
                
                self.memIdx = list(range(count))
            except:
                self.memIdx = [0]

            self.perm = 0

        elif dialogue == 'new':
            self.dialogue = 'prompts'
            self.perm = 1

        embed = discord.Embed(title=f"char settings: {self.char}", description=f'> index can be any within the list' if not self.perm else f'> index can be None', color=discord.Color.green())
        embed.add_field(name="index(ls)", value=f"{self.memIdx}", inline=False)
        embed.add_field(name="path", value=f"{self.dialogue}", inline=True)
        embed.add_field(name="reply_lang", value=f"{self.reply_lang}", inline=True)
        await ctx.send(embed=embed)
        self.char_is_set = True

    @cmds.hybrid_command(description=f'please remind that the command is still WIP')
    async def tchat(self, ctx, index: int = None, reply_lang: Optional[Literal['th', 'en']] = None):
        if self.char_is_set:
            self.user_id = ctx.author.id
            voice_channel = ctx.author.voice.channel
            if self.voice and self.voice.is_connected():
                await self.voice.move_to(voice_channel)
            else:
                self.voice = await voice_channel.connect()

            await ctx.send("``` voice chat on ```")

            self.rdy = 1
            self.payload = ctx
            self.reply_lang = reply_lang
                
            if self.perm:
                mount = '.txt'
            else:
                if index != None:
                    mount = f'/conversation_{index}.txt'
                else:
                    embed = discord.Embed(title=f"ERR: index not found", description="> as you've chosen 'con', the index is required", color=discord.Color.red())
                    return await ctx.send(embed=embed)

            with open(f'{self.dialogue}/{self.char}{mount}', "r", encoding='utf-8') as file:
                mode = file.read()

            if self.perm:
                self.messages  = [{"role": "system", "content": f"{mode}"}]
                self.suffix = self.getSuffix()
            else:
                self.messages = json.loads(mode)
                self.suffix = index

        else:
            embed = discord.Embed(title=f"ERR: tchat is not set", description="> it seems like you've not set the character yet", color=discord.Color.red())
            embed.add_field(name="follow the process", value=f"try to use /set_char char:<in the desc> path:<choices>", inline=False)
            await ctx.send(embed=embed)

    @cmds.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if self.user_id is not None and self.rdy:
            #bot itself changed
            if member == self.bot.user:
                if not after.channel:
                    self.voice = None
            #user changed
            elif member.id == self.user_id:
                if before.channel != after.channel:
                    if self.voice:
                        await self.voice.disconnect()
                    if after.channel:
                        voice_channel = await self.bot.fetch_channel(after.channel.id)
                        self.voice = await voice_channel.connect()
                        print(f"Joined {after.channel}")

async def setup(bot):
    await bot.add_cog(Tchat(bot))