import os

import discord
import openai
import functools
import typing
import asyncio
from dotenv import load_dotenv
import datetime
import re

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')

openai.api_key = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)

emotes_dict = {
    '751668390455803994': '(Cat smiling and crying and putting thumb up emote)',
    '751668390455803995': '(Cat smiling and crying and putting thumb down emote)',
    'wuuuaaat': '(Usada Pekora saying "whattttttt!!!?" emote)',
    'urrrrrrrr':'(Usada Pekora being frustrated emote)',
    '602092952159780874': '(huh? emote)',
    '884491552594985080': '("brain not working" emote)',
    'Abbbbbbb': '(dab emote)',
    'CuteQuestion':'(person having a question emote)',
    'Waow': '(Wow! emote)',
    'xdw':'(xD emote)',
    'xd':'(xD emote)',
    'Deb':'(fox girl dabbing emote)',
    'toad':'(toad dancing emote)',
}

gpt_versions = [
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-16k-0613',
    'gpt-4',
    'gpt-4-0314',
    'gpt-4-0613',
]
# default_response = [
#     {'role' : 'assistant', 'content' : '[GPT]: Hello there! How may I assist you?\n\n[zui-botto]: Hi! What can I help you with today? <:751668390455803994:955646373246672966>'}, 
#     {'role' : 'assistant', 'content' : '[GPT]: How can I assist you today?\n\n[zui-botto]: Hello! Is there anything I can help you with today? <:751668390455803994:955646373246672966>'}
#     ]

default_response = []

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

def init_intent():
    return {
        'chat_gpt' : True,
        'model' : 'gpt-3.5-turbo',
        'chat_msgs' : default_response,
        'command_txt' : {'role' : 'system', 'content' : ''},
        'current_token' : 0,
        'token_limit' : 3000
    }

def format_msg(role: str, msg: str):
    return {"role": role, "content": msg}

def contains_botto(message):
    return message.find('[zui-botto]:')

@to_thread
def generate(model, messages):
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=0.2)

def find_gpt_botto(message):
    gpt_indices = []
    botto_indices = []

    i = message.find('[zui-botto]')
    while i != -1:
        botto_indices.append(i)
        i = message.find('[zui-botto]', i + 1)

    i = message.find('[GPT]')
    while i != -1:
        gpt_indices.append(i)
        i = message.find('[GPT]', i + 1)

    chat_indicies = []
    for botto_idx in botto_indices:
        chat_indicies.append((botto_idx, 'zui-botto'))

    for gpt_idx in gpt_indices:
        chat_indicies.append((gpt_idx, 'gpt'))

    chat_indicies.sort()
    if len(botto_indices) != 1 or len(gpt_indices) != 1:
        correction_needed = True
    else:
        if botto_indices[0] < gpt_indices[0]:
            correction_needed = True
    return gpt_indices, botto_indices

def get_gpt_botto_msg(message, gpt_indices, botto_indices):
    gpt_messages = []
    botto_mesages = []


@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break

    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )

channel_intents = {}

@client.event
async def on_message(message):
    if message.author.bot:
            return
    try:
        guild_id = message.guild.id
        if not ('bot' in str(message.channel) or 'gpt' in str(message.channel) or 'chat' in str(message.channel)):
            return
    except:
        guild_id = 'dm'
    channel = message.channel
    channel_id = message.channel.id

    intent_id = str(channel_id) + '_' + str(guild_id)
    global channel_intents
    if intent_id not in channel_intents:
        channel_intents[intent_id] = init_intent()

    print(f'{message.author}: {message.content}')
    if len(message.content) > 0:
        if message.content[0] == '!':
            if len(message.content) > 1:
                cmd = message.content[1:]
                if cmd == 'newchat':
                    embedVar = discord.Embed(title="New chat started!", color=0x74a89b)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                    channel_intents[intent_id]['chat_gpt'] = True
                    channel_intents[intent_id]['chat_msgs'] = []
                elif cmd == 'stopchat':
                    embedVar = discord.Embed(title="Chat stopped!", color=0xf75948)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                    channel_intents[intent_id]['chat_gpt'] = False
                    channel_intents[intent_id]['chat_msgs'] = []

                elif cmd == '!' or cmd == '!!' or cmd == '!!!' or cmd == '!!!!':
                    await channel.send('!' + cmd, reference=message, mention_author=False)

                elif cmd == 'currenttoken':
                    embedVar = discord.Embed(title=f"Tokens used : {channel_intents[intent_id]['current_token']}", color=0xf75948)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)

                elif cmd == 'currentlimit':
                    embedVar = discord.Embed(title=f"Current token limit : {channel_intents[intent_id]['token_limit']}", color=0xf75948)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)

                elif cmd == 'ping':
                    await channel.send('Pong! {0}ms'.format(round(client.latency*1000, 1)), reference=message, mention_author=False)

                elif cmd[:16] == 'set systemprompt':
                    prompt = cmd[17:]
                    if prompt == 'default':
                        channel_intents[intent_id]['command_txt'] = {'role' : 'system', 'content' : DEFAULT_PROMPT}
                    else: channel_intents[intent_id]['command_txt'] = {'role' : 'system', 'content' : prompt}
                    channel_intents[intent_id]['chat_msgs'] = []

                    embedVar = discord.Embed(title=f"System prompt set as : {prompt[:220]}", color=0x22f5dc)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                elif cmd[:15] == 'set gpt-version':
                    version = cmd[16:]

                    if not version in gpt_versions:
                        embedVar = discord.Embed(title=f"Invalid GPT version", description=f"Available versions: \n{str(gpt_versions)}", color=0xf75948)
                        await channel.send(embed=embedVar, reference=message, mention_author=False)
                    else:
                        channel_intents[intent_id]['model'] = version
                        embedVar = discord.Embed(title=f"Set GPT version as : {version[:220]}", color=0x22f5dc)
                        await channel.send(embed=embedVar, reference=message, mention_author=False)

                elif cmd == 'currentsystemprompt':
                    prompt = channel_intents[intent_id]['command_txt']['content']

                    embedVar = discord.Embed(title=f"Current system prompt : {prompt[:220]}", color=0x22f5dc)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                # elif cmd == 'help':
                #     embedVar = discord.Embed(title="Commands List:", color=0x22f5dc)
                #     embedVar.add_field(name="ChatGPT Commands", value="`!newchat` to start a new conversation, use the '` key' before any messages to talk to the bot.", inline=False)
                #     await channel.send(embed=embedVar)
        elif message.content[0] == '`' or guild_id == 'dm':
            if len(message.content) > 1 or guild_id == 'dm':
                if guild_id != 'dm':
                    cmd = message.content[1:]
                else:
                    cmd = message.content
                if channel_intents[intent_id]['chat_gpt'] and cmd != 'needsmorejpeg' and cmd != 'triggered':
                    msg = cmd
                    for key in emotes_dict.keys():
                        msg = re.sub(key, emotes_dict[key], msg)
                    msg = re.sub(r"<:\(", "(", msg)
                    msg = re.sub(":[0-9]*>", "", msg)

                    current_chat_msgs = channel_intents[intent_id]['chat_msgs'] + [{'role' : 'user', 'content' : msg}]
                    loading_msg = await channel.send('<a:pluzzlel:959061506568364042>', reference=message, mention_author=False)


                    updated_system_prompt = channel_intents[intent_id]['command_txt']
                    # updated_system_prompt['content'] += ' For reference, the current time is ' + str(datetime.datetime.utcnow())

                    error = False
                    try:
                        response = await generate(model=channel_intents[intent_id]['model'], messages=[updated_system_prompt] + channel_intents[intent_id]['chat_msgs'] + [{'role' : 'user', 'content' : msg}])
                    except Exception as e:
                        channel_intents[intent_id]['chat_msgs'] = channel_intents[intent_id]['chat_msgs'][2:]
                        embedVar = discord.Embed(title=f"{e}", color=0xf75948)
                        await channel.send(embed=embedVar, reference=message, mention_author=False)
                        error = True
                        await loading_msg.delete()

                    if not error:
                        response_msg = response.choices[0].message.content
                        used_tokens = response.usage.total_tokens
                        channel_intents[intent_id]['current_token'] = used_tokens

                        print(f'ChatGPT Response : {response_msg}')
                        
                        if used_tokens > channel_intents[intent_id]['token_limit']:
                            channel_intents[intent_id]['chat_msgs'] = channel_intents[intent_id]['chat_msgs'][2:]

                        channel_intents[intent_id]['chat_msgs'].append({'role' : 'user', 'content' : msg})
                        channel_intents[intent_id]['chat_msgs'].append({'role' : 'assistant', 'content' : response_msg})

                        # gpt_indices, botto_indices = find_gpt_botto(response_msg)

                        # i = response_msg.find('[zui-botto]')
                        # while i != -1:
                        #     botto_indices.append(i)
                        #     i = response_msg.find('[zui-botto]', i + 1)

                        # i = response_msg.find('[GPT]')
                        # while i != -1:
                        #     gpt_indices.append(i)
                        #     i = response_msg.find('[GPT]', i + 1)

                        # chat_indicies = []
                        # for botto_idx in botto_indices:
                        #     chat_indicies.append((botto_idx, 'zui-botto'))

                        # for gpt_idx in gpt_indices:
                        #     chat_indicies.append((gpt_idx, 'gpt'))

                        # chat_indicies.sort()
                        # if len(botto_indices) != 1 or len(gpt_indices) != 1:
                        #     correction_needed = True
                        # else:
                        #     if botto_indices[0] < gpt_indices[0]:
                        #         correction_needed = True

                        # if len(botto_indices) == 0:
                        #     gpt_outputs = []
                        #     for idx in range(len(gpt_indices) - 1):
                        #         gpt_output.append(response_msg[gpt_outputs[idx] + 5:gpt_outputs[idx+1]])
                        #     gpt_out_string = "\n\n".join(gpt_outputs)

                        #     uncorrected_chat_msgs = current_chat_msgs + [{'role' : 'user', 'content' : msg}] + [{'role' : 'assistant', 'content' : response_msg}]
                        #     uncorrected_chat_msgs.append({'role' : 'user', 'content' : 'Stay in character! Please continue as zui-botto and follow the format in those sentences'})
                        #     correction_response = await generate(model=channel_intents[intent_id]['model'], messages=[channel_intents[intent_id]['command_txt']] + uncorrected_chat_msgs)

                        #     correction_response_msg = correction_response.choices[0].message.content
                        #     correction_fixed_msg = '[GPT]: ' + gpt_out_string + '\n\n' + '[zui-botto]: ' + correction_response_msg
                        #     channel_intents[intent_id]['chat_msgs'].append({'role' : 'assistant', 'content' : correction_response_msg})
                        #     print(f'Correction Response : {correction_response_msg}')
                        try:
                            if updated_system_prompt['content'].find('zui-botto') != -1:

                                botto_idx = contains_botto(response_msg)
                                if botto_idx == -1:
                                    gpt_idx = response_msg.find('[GPT]:')

                                    if gpt_idx == -1:
                                        pass
                                    else:
                                        response_msg = response_msg[gpt_idx + 7:]

                                    updated_system_prompt = channel_intents[intent_id]['command_txt']
                                    updated_system_prompt['content'] += ' For reference, the current time is ' + str(datetime.datetime.utcnow())

                                    channel_intents[intent_id]['chat_msgs'].append({'role' : 'user', 'content' : 'Stay in character! Please continue as zui-botto and follow the format in those sentences'})
                                    completed = False
                                    while not completed:
                                        try:
                                            correction_response = await generate(model=channel_intents[intent_id]['model'], messages=[updated_system_prompt] + channel_intents[intent_id]['chat_msgs'])
                                            completed = True
                                        except:
                                            channel_intents[intent_id]['chat_msgs'] = channel_intents[intent_id]['chat_msgs'][2:]
                                    correction_response_msg = correction_response.choices[0].message.content
                                    channel_intents[intent_id]['chat_msgs'].append({'role' : 'assistant', 'content' : correction_response_msg})
                                    print(f'Correction Response : {correction_response_msg}')

                                else:
                                    gpt_idx = response_msg.find('[GPT]:')
                                    if gpt_idx == -1:
                                        updated_system_prompt = channel_intents[intent_id]['command_txt']
                                        updated_system_prompt['content'] += ' For reference, the current time is ' + str(datetime.datetime.utcnow())

                                        channel_intents[intent_id]['chat_msgs'].append({'role' : 'user', 'content' : 'Stay in character! Please continue as zui-botto and follow the format in those sentences'})
                                        completed = False
                                        while not completed:
                                            try:
                                                correction_response = await generate(model=channel_intents[intent_id]['model'], messages=[updated_system_prompt] + channel_intents[intent_id]['chat_msgs'])
                                                completed = True
                                            except:
                                                channel_intents[intent_id]['chat_msgs'] = channel_intents[intent_id]['chat_msgs'][2:]
                                        correction_response_msg = correction_response.choices[0].message.content
                                        channel_intents[intent_id]['chat_msgs'].append({'role' : 'assistant', 'content' : '[GPT]:' + correction_response_msg[13:] + '\n\n' + correction_response_msg})
                                        print(f'Correction Response : {correction_response_msg}')

                                    response_msg = response_msg[botto_idx + 13:]

                                restart = False
                                attempts = 0
                                if response_msg.find('[FILTERING]') != -1:
                                    channel_intents[intent_id]['chat_msgs'] = []
                                    response_msg = '<:751668390455803995:942910387282673684>'
                                else:
                                    while response_msg.find(', ChatGPT.') != -1 or response_msg.find(', ChatGPT!') != -1 or response_msg.find(', ChatGPT?') != -1 \
                                        or response_msg.find('ChatGPT!') != -1 or response_msg.find('ChatGPT?') != -1 \
                                        or response_msg.find('[zui-botto]') != -1 or response_msg.find('[GPT]') != -1 or botto_idx < 4:
                                        attempts = attempts + 1
                                        restart = True
                                        if len(channel_intents[intent_id]['chat_msgs']) == 2:
                                            channel_intents[intent_id]['chat_msgs'] = []
                                            response = await generate(model=channel_intents[intent_id]['model'], messages=[updated_system_prompt] + channel_intents[intent_id]['chat_msgs'] + [{'role' : 'user', 'content' : msg}])
                                            response_msg = response.choices[0].message.content
                                        else:
                                            channel_intents[intent_id]['chat_msgs'] = []
                                            response = await generate(model=channel_intents[intent_id]['model'], messages=[updated_system_prompt] + channel_intents[intent_id]['chat_msgs'] + [{'role' : 'user', 'content' : msg}])
                                            response_msg = response.choices[0].message.content
                                        if attempts > 1:
                                            channel_intents[intent_id]['chat_msgs'] = []
                                            response_msg = '<:751668390455803994:955646373246672966>'
                                            break

                                if restart and attempts <= 1:
                                    botto_idx = contains_botto(response_msg)
                                    # response_msg = response_msg[botto_idx + 13:]
                                    channel_intents[intent_id]['chat_msgs'] = []
                                    response_msg = '<:751668390455803994:955646373246672966>'
                        except e:
                            channel_intents[intent_id]['chat_msgs'] = []
                            response_msg = '<:751668390455803994:955646373246672966>'
                            embedVar = discord.Embed(title=f"{e}", color=0x22f5dc)
                            await channel.send(embed=embedVar, reference=message, mention_author=False)
                        await loading_msg.delete()

                        # response_msg = response_msg + '(<@315763195727970305> <@315763195727970305> <@315763195727970305>) <- this is me doing it, not the bot'

                        msg_list = response_msg.split('\n')
                        num_char = 0
                        chunk_list =[]
                        for message_chunk in msg_list:
                            while(len(message_chunk) > 1990):
                                await channel.send(message_chunk[:1990], reference=message, mention_author=False)
                                message_chunk = message_chunk[1990:]

                            chunk_list.append(message_chunk)
                            num_char  = num_char + len(message_chunk)
                            if num_char > 1990 - len(chunk_list):
                                chunk_list = chunk_list[:-1]
                                num_char  = num_char - len(message_chunk)
                                await channel.send("\n".join(chunk_list), reference=message, mention_author=False)

                                num_char  = len(message_chunk)
                                chunk_list = [message_chunk]
                        await channel.send("\n".join(chunk_list), reference=message, mention_author=False)

                        # if len(response_msg) > 1900:
                        #     msg_list = response_msg.split('/n')
                        #     for message in msg_list:
                        #         await channel.send(message)
                        # else:
                        #     await channel.send(response_msg)


client.run(TOKEN)