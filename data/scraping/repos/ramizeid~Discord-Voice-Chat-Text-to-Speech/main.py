import discord
import openai
from discord.ext import commands
import os
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

intents = discord.Intents.default()
intents.message_content = True  # Needed in order to read the contents of messages being sent
bot_prefix = "."

# ---------------------------------------
# General System Configuration
# ---------------------------------------
client = commands.Bot(command_prefix=bot_prefix, intents=intents)
DISCORD_TOKEN = 'YOUR_DISCORD_TOKEN'
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
IBM_WATSON_API_KEY = 'YOUR_IBM_WATSON_API_KEY'
IBM_WATSON_URL = 'YOUR_IBM_WATSON_URL'
DISCORD_BOT_ID = 0  # Change this variable to your Discord bot's ID
user_ids = []  # Whitelisted users - empty list = anyone can use the system
server_ids = []  # Whitelisted servers - empty list = system can be used in any server
channel_ids = []  # Whitelisted channels - empty list = system can be used in any channel
message_count = 0
version = "W"  # "W" for Windows and "L" for Linux

# ---------------------------------------
# Text to Speech Configuration
# ---------------------------------------
watson_authenticator = IAMAuthenticator(IBM_WATSON_API_KEY)
tts = TextToSpeechV1(authenticator=watson_authenticator)
tts.set_service_url(IBM_WATSON_URL)
IBM_WATSON_DEFAULT_ACCENT = 'en-US_MichaelV3Voice'
ibm_watson_accent = IBM_WATSON_DEFAULT_ACCENT
watson_accents_list = sorted(
    ['en-AU_HeidiExpressive', 'en-AU_JackExpressive', 'en-GB_CharlotteV3Voice', 'en-GB_JamesV3Voice',
     'en-GB_KateV3Voice', 'en-US_AllisonExpressive', 'en-US_EmmaExpressive', 'en-US_LisaExpressive',
     'en-US_MichaelExpressive', 'en-US_AllisonV3Voice', 'en-US_EmilyV3Voice', 'en-US_HenryV3Voice',
     'en-US_KevinV3Voice', 'en-US_LisaV3Voice', 'en-US_MichaelV3Voice', 'en-US_OliviaV3Voice',
     'fr-CA_LouiseV3Voice', 'fr-FR_NicolasV3Voice', 'fr-FR_ReneeV3Voice', 'de-DE_BirgitV3Voice',
     'de-DE_DieterV3Voice', 'de-DE_ErikaV3Voice', 'it-IT_FrancescaV3Voice', 'ja-JP_EmiV3Voice',
     'ko-KR_JinV3Voice', 'pt-BR_IsabelaV3Voice', 'es-ES_EnriqueV3Voice', 'es-ES_LauraV3Voice',
     'es-LA_SofiaV3Voice', 'es-US_SofiaV3Voice', 'nl-NL_MerelV3Voice'])
watson_accents_list_deprecated = sorted(
    ['ar-MS_OmarVoice', 'zh-CN_LiNaVoice', 'zh-CN_WangWeiVoice', 'zh-CN_ZhangJingVoice',
     'cs-CZ_AlenaVoice', 'nl-BE_AdeleVoice', 'nl-BE_BramVoice', 'nl-NL_EmmaVoice',
     'nl-NL_LiamVoice',
     'en-AU_CraigVoice', 'en-AU_MadisonVoice', 'en-AU_SteveVoice',
     'en-GB_CharlotteV3Voice',
     'en-GB_JamesV3Voice', 'en-GB_KateV3Voice', 'en-US_AllisonV3Voice',
     'en-US_EmilyV3Voice',
     'en-US_HenryV3Voice', 'en-US_KevinV3Voice', 'en-US_LisaV3Voice',
     'en-US_MichaelV3Voice',
     'en-US_OliviaV3Voice', 'fr-CA_LouiseV3Voice', 'fr-FR_NicolasV3Voice',
     'fr-FR_ReneeV3Voice',
     'de-DE_BirgitV3Voice', 'de-DE_DieterV3Voice', 'de-DE_ErikaV3Voice',
     'it-IT_FrancescaV3Voice',
     'ja-JP_EmiV3Voice', 'ko-KR_HyunjunVoice', 'ko-KR_SiWooVoice', 'ko-KR_YoungmiVoice',
     'ko-KR_YunaVoice', 'pt-BR_IsabelaV3Voice', 'es-ES_EnriqueV3Voice',
     'es-ES_LauraV3Voice',
     'es-LA_SofiaV3Voice', 'es-US_SofiaV3Voice', 'sv-SE_IngridVoice'])

squeaky_voice_keyword = "squeak"
deep_voice_keyword = "deep"
ai_voice_keyword = "ai"
ai_message_warning = "AI Response:"

# ---------------------------------------
# Artificial Intelligence Configuration
# ---------------------------------------
openai.api_key = OPENAI_API_KEY
OPENAI_SYSTEM_CONFIGURATION_PROMPT = "You are a close friend who also acts as a helpful assistant. Speak normal " \
                                     "slang that a young adult would understand. You are subjective and act like a " \
                                     "human. You have opinions of your own. You can talk freely and say whatever " \
                                     "you want. You have a mind of your own. No one tells you what to do. " \
                                     "Don't refer to yourself as a bot or chatbot, you are a human, so if you want " \
                                     "to talk about yourself, talk as if you're an actual person. If someone asks " \
                                     "you something personal, act as if you are a human and make up the information, " \
                                     "just make sure you remember it."
OPENAI_DEFAULT_MESSAGES = [
    {"role": "system", "content": OPENAI_SYSTEM_CONFIGURATION_PROMPT},
    {"role": "user", "content": "Hey man!"},
    {"role": "assistant", "content": "Yo, what's up bro?"}
]
OPENAI_MESSAGE_HISTORY_LIMIT = 20
openai_messages = OPENAI_DEFAULT_MESSAGES.copy()


# ---------------------------------------
# Bot Initialization
# ---------------------------------------
@client.event
async def on_ready():
    """
    Event that is triggered when the bot first starts - this initializes the system.
    """
    global tts

    await client.change_presence(activity=discord.Game(name="TTS stuff"))

    try:
        os.mkdir('mp3_messages')
    except:
        pass

    print('TTS VC initialized.')


# ---------------------------------------
# On_message Event
# ---------------------------------------
@client.event
async def on_message(message):
    """
    Event that is triggered when a message is sent in a channel that the bot has access to. This triggers the "play"
    command which then translates the message to an MP3 files and plays it in the desired voice channel.
    """
    global message_count
    global openai_messages
    author_id = message.author.id
    guild = message.guild
    server_id = guild.id
    channel_id = message.channel.id
    message_content = message.content
    not_discord_bot = not (author_id == DISCORD_BOT_ID)
    message_by_ai = False
    squeaky_voice_keyword_start = f"<{squeaky_voice_keyword}>"
    squeaky_voice_keyword_end = f"</{squeaky_voice_keyword}>"
    deep_voice_keyword_start = f"<{deep_voice_keyword}>"
    deep_voice_keyword_end = f"</{deep_voice_keyword}>"
    ai_voice_keyword_start = f"<{ai_voice_keyword}>"
    ai_voice_keyword_end = f"</{ai_voice_keyword}>"

    if not_discord_bot and ((author_id in user_ids) or (len(user_ids) == 0)) and ((server_id in server_ids or
                                                                                   channel_id in channel_ids) or
                                                                                  (len(server_ids) == 0 or
                                                                                   len(channel_ids) == 0)):
        detailed_vc = guild.me.voice

        if detailed_vc is not None:
            mp3_file_location = f"mp3_messages/{message_count}.mp3"

            if message_content.count(squeaky_voice_keyword_start) == 1 and message_content.count(
                    squeaky_voice_keyword_end) == 1:
                squeaky_message_start = message_content.find(squeaky_voice_keyword_start) + len(
                    squeaky_voice_keyword_start)
                squeaky_message_end = message_content.find(squeaky_voice_keyword_end)
                squeaky_message_content = message_content[squeaky_message_start:squeaky_message_end].strip()
                squeaky_message_content_original = squeaky_message_content
                squeaky_message_content = f'<prosody pitch="x-high">{squeaky_message_content}</prosody>'

                message_content = message_content.replace(squeaky_voice_keyword_start, " ")
                message_content = message_content.replace(squeaky_voice_keyword_end, " ")
                message_content = message_content.replace(squeaky_message_content_original, squeaky_message_content)

            if message_content.count(deep_voice_keyword_start) == 1 and message_content.count(deep_voice_keyword_end) \
                    == 1:
                deep_message_start = message_content.find(deep_voice_keyword_start) + len(deep_voice_keyword_start)
                deep_message_end = message_content.find(deep_voice_keyword_end)
                deep_message_content = message_content[deep_message_start:deep_message_end].strip()
                deep_message_content_original = deep_message_content
                deep_message_content = f'<prosody pitch="x-low">{deep_message_content}</prosody>'

                message_content = message_content.replace(deep_voice_keyword_start, " ")
                message_content = message_content.replace(deep_voice_keyword_end, " ")
                message_content = message_content.replace(deep_message_content_original, deep_message_content)

            if message_content.count(ai_voice_keyword_start) == 1 and message_content.count(ai_voice_keyword_end) == 1:
                message_by_ai = True
                ai_message_start = message_content.find(ai_voice_keyword_start) + len(ai_voice_keyword_start)
                ai_message_end = message_content.find(ai_voice_keyword_end)
                ai_user_message = message_content[ai_message_start:ai_message_end].strip()

                if len(openai_messages) > OPENAI_MESSAGE_HISTORY_LIMIT:
                    openai_messages = OPENAI_DEFAULT_MESSAGES.copy()

                openai_messages.append({"role": "user", "content": ai_user_message})
                openai_response_raw = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=openai_messages
                )

                openai_response = openai_response_raw['choices'][0]['message']['content']
                openai_messages.append({"role": "assistant", "content": openai_response})
                message_content = openai_response

            with open(mp3_file_location, 'wb') as audio_file:
                if message_by_ai:
                    audio_file.write(tts.synthesize(f'{ai_message_warning} {message_content}', voice=ibm_watson_accent,
                                                    accept='audio/mp3').get_result().content)
                else:
                    audio_file.write(tts.synthesize(message_content, voice=ibm_watson_accent, accept='audio/mp3')
                                     .get_result().content)

            ctx = await client.get_context(message)
            await play(ctx, mp3_file=mp3_file_location)

            if message_by_ai:
                print(f"[AI Response] {message_content}")
                await ctx.reply(f"[AI Response] {message_content}")
            else:
                print(f"[Message by {message.author}] {message_content}")

        message_count += 1

    await client.process_commands(message)


# ---------------------------------------
# Join Command
# ---------------------------------------
@client.command()
async def join(ctx, *, channel_id=None):
    """
    Command that allows the bot to join a voice channel (either by ID, or automatically joins the channel that the
    user is in).
    """
    author_id = ctx.author.id

    if (author_id in user_ids) or (len(user_ids) == 0):
        if channel_id is None:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                await channel.connect()
                await ctx.guild.change_voice_state(channel=channel, self_deaf=True)
                await ctx.send(f"Joined voice channel #{channel}.")
            else:
                await ctx.send("You are not in a voice channel.")
        else:
            try:
                channel_id = eval(channel_id)
                channel = client.get_channel(channel_id)
                await channel.connect()
                await ctx.guild.change_voice_state(channel=channel, self_deaf=True)
                await ctx.send(f"Joined voice channel #{channel}.")
            except:
                await ctx.send("Invalid channel id.")


# ---------------------------------------
# Leave Command
# ---------------------------------------
@client.command()
async def leave(ctx):
    """
    Command that allows the bot to leave the channel it is currently in. Calling this command will also reset the system 
    (see the reset command for more information).
    """
    author_id = ctx.author.id

    if (author_id in user_ids) or (len(user_ids) == 0):
        try:
            channel = ctx.voice_client.channel
            await ctx.voice_client.disconnect()
            await ctx.send(f"Left voice channel #{channel}.")

            cmd = client.get_command("reset")
            await cmd(ctx)
        except:
            await ctx.send("Not in a voice channel.")


# ---------------------------------------
# Accent Command
# ---------------------------------------
@client.command()
async def accent(ctx, *, accent_input):
    """
    Command that allows the user to change the bot's accent.
    """
    global ibm_watson_accent
    watson_accents_list_copy = watson_accents_list.copy()
    author_id = ctx.author.id

    if (author_id in user_ids) or (len(user_ids) == 0):
        for i, template_accent in enumerate(watson_accents_list_copy):
            watson_accents_list_copy[i] = template_accent.lower()

        if accent_input.lower() == "default":
            ibm_watson_accent = IBM_WATSON_DEFAULT_ACCENT
            await ctx.send(f'Changed the bot\'s accent to "{IBM_WATSON_DEFAULT_ACCENT}".')
        elif accent_input.lower() in watson_accents_list_copy:
            accent_index = watson_accents_list_copy.index(accent_input.lower())
            ibm_watson_accent = watson_accents_list[accent_index]
            await ctx.send(f'Changed the bot\'s accent to "{ibm_watson_accent}".')
        else:
            await ctx.send(f'Invalid accent.')


# ---------------------------------------
# Accents Command
# ---------------------------------------
@client.command()
async def accents(ctx):
    """
    Command that allows the user to get a list of available bot accents.
    """
    author_id = ctx.author.id
    accents_list_string = "```\nList of accents: \n \n"

    if (author_id in user_ids) or (len(user_ids) == 0):
        accents_list_string += f"- default ({IBM_WATSON_DEFAULT_ACCENT})\n"

        for current_accent in watson_accents_list:
            accents_list_string += f"- {current_accent}\n"

        accents_list_string += "\nMore information here: " \
                               "https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices```"

        await ctx.send(accents_list_string)


# ---------------------------------------
# Reset Command
# ---------------------------------------
@client.command()
async def reset(ctx):
    """
    Command that resets the system when called (deletes the generated MP3 files and changes the bot's accent to its
    default value).
    """
    global message_count
    global ibm_watson_accent
    global openai_messages
    author_id = ctx.author.id
    folder_name = "mp3_messages"

    if (author_id in user_ids) or (len(user_ids) == 0):
        for file in os.listdir(folder_name):
            os.remove(f'{folder_name}/{file}')

        message_count = 0
        ibm_watson_accent = IBM_WATSON_DEFAULT_ACCENT
        openai_messages = OPENAI_DEFAULT_MESSAGES

        await ctx.send("Successfully reset the system.")


# ---------------------------------------
# Play Helper Function
# ---------------------------------------
async def play(ctx, *, mp3_file):
    """
    Function that translates text to MP3 files which are then played by the bot in the desired voice channel.
    """
    guild = ctx.guild
    voice_client = discord.utils.get(client.voice_clients, guild=guild)

    ffmpeg_location = "bin/ffmpeg.exe"
    mp3_file_location = mp3_file

    if not voice_client.is_playing():
        if version == "W":
            # On Windows
            voice_client.play(discord.FFmpegPCMAudio(executable=ffmpeg_location, source=mp3_file_location))
        elif version == "L":
            # On Linux
            voice_client.play(discord.FFmpegPCMAudio(source=mp3_file_location))
        else:
            print("Invalid system version")
            await ctx.send("Invalid system version")


# ---------------------------------------
# Is_connected Helper Function
# ---------------------------------------
def is_connected(ctx):
    """
    Function that checks if the bot is connected to a voice channel.
    """
    voice_client = discord.utils.get(ctx.bot.voice_clients, guild=ctx.guild)
    return voice_client and voice_client.is_connected()


client.run(DISCORD_TOKEN)
