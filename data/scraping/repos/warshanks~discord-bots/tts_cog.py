from os import makedirs
import discord
import openai
from elevenlabs import set_api_key, generate, save
from discord import FFmpegOpusAudio
from discord.ext import commands
from cogs.chat_cog import generate_response
from config import openai_token, openai_org, elevenlabs_token, vcs

makedirs("./speech", exist_ok=True)

# Set OpenAI API key and organization
openai.api_key = openai_token
openai.organization = openai_org

bot = commands.Bot(command_prefix="~", intents=discord.Intents.all())

set_api_key(elevenlabs_token)


def write_audio_to_file(filename, tts_response):
    with open(filename, "wb") as out:
        out.write(tts_response)


# noinspection PyShadowingNames
class TTSCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.FFMPEG_OPTIONS = {'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5',
                               'options': '-vn'}
        self.vc = None

    @commands.Cog.listener()
    async def on_message(self, message):
        # Ignore messages from the bot itself,
        # from the system user,
        # or from channels other than the designated one,
        # or that start with '!'
        if (message.author.bot or
                message.author.system or
                message.channel.id not in vcs or
                message.content.startswith('!')):
            return

        async with message.channel.typing():
            openai_model = 'gpt-3.5-turbo'
            conversation_log = [{'role': 'system', 'content':
                                 'You are a friendly secretary named KC.'}]

            ttt_response = await generate_response(
                message,
                conversation_log,
                openai_model
            )

            try:
                tts_response = generate(
                    text=ttt_response,
                    api_key=elevenlabs_token,
                    voice="Rachel",
                    model="eleven_monolingual_v1"
                )
                save(tts_response, "./speech/output.mp3")
                print('Audio content written to file "./speech/output.mp3"')

                await message.reply(ttt_response)

                tts_output = await FFmpegOpusAudio.from_probe("./speech/output.mp3")

                vc.play(tts_output)
            except Exception as e:
                await message.reply(e)

    # noinspection PyGlobalUndefined
    @bot.tree.command(name='join', description='Join the voice channel')
    async def join(self, ctx):
        global vc
        # Defer the response to let the user know that the bot is working on the request
        # noinspection PyUnresolvedReferences
        await ctx.response.defer(thinking=True, ephemeral=True)
        voice_channel = ctx.user.voice.channel
        vc = await voice_channel.connect()
        await ctx.followup.send("Joined!", ephemeral=True)

    @bot.tree.command(name='tts-kick', description='Leave the voice channel')
    async def tts_kick(self, ctx):
        await ctx.response.defer(thinking=True, ephemeral=True)
        await vc.disconnect()
        await ctx.followup.send("Left!", ephemeral=True)
