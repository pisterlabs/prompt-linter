from twitchAPI import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.types import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import asyncio
import vlc
from dotenv import load_dotenv
import os
from BlenderChatbot import ChatBot
from OpenAIChatbot import OpenAIChatbot
from tts import TTS
# local tts is currently not working
# from localtts import local_speak
from stt import listen, start_listen_thread
import subprocess
import json
import sqlite3
import random
import keyboard
import time
# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def index():
#     asyncio.run(twitch_connect())
#     print("its working! its working!")
#     return "Hello World"

# this is to be able to use the .env file in the same directory
load_dotenv()

# load in the configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

# set up the authentication stuff
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TARGET_CHANNEL = config['target_channel']
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
engagement_mode = config['engagement_mode']
tts = config['tts']
stt = config['stt']
bot_name = config['bot_name']
local_tts = config['local_tts']
push_to_talk = config['push_to_talk']

# initialize the bot, change one of the available options in config to true

if config['OpenAI']:
    ai = OpenAIChatbot(bot_name)
elif config['blenderbot']:
    ai = ChatBot() 
# currently not working
# elif config['Dolly']:
#     ai = DollyChatbot()

text_to_speech = TTS()

engagement_list = {} # to keep track of users and if they are engaged

try:
    conn = sqlite3.connect('app.db')

    # looks at the database and checks if the users table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create the users table if it doesn't exist
        conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, points INTEGER)')
        conn.execute('CREATE TABLE mods (id INTEGER PRIMARY KEY, name TEXT)')
        conn.execute('CREATE TABLE quotes (id INTEGER PRIMARY KEY, quote TEXT, author TEXT)')
        conn.execute('INSERT INTO mods (name) VALUES (?)', (TARGET_CHANNEL,))
        conn.execute('INSERT INTO mods (name) VALUES (?)', (bot_name,))
        conn.commit()

    messages_table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")

    if not messages_table:
        conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, user_message TEXT, author TEXT, timestamp TEXT, chatbot_response TEXT)')
        conn.commit()

except sqlite3.Error as e:
    print(e)
finally:
    if conn:
        conn.close() 

# checks to see if the credentials file exists, if it does, load it
try:
    with open('user_credentials.json', 'r') as f:
        user_credentials = json.load(f)
        print("User credentials loaded")
except FileNotFoundError:
    print("User credentials not found")
    user_credentials = None

async def on_ready(ready_event: EventData):
    await ready_event.chat.join_room(TARGET_CHANNEL)

async def on_message(msg: ChatMessage):
    # if user does not exist in database, add them
    # if user does exist, increment their points
    # if user is a mod, they have full control over other users' points

    conn = sqlite3.connect('app.db')
    messageUser = conn.execute('SELECT * FROM users WHERE name = ?', (msg.user.name,)).fetchone()
    if messageUser is None:
        conn.execute('INSERT INTO users (name, points) VALUES (?, ?)', (msg.user.name, 10))

    if conn.execute('SELECT * FROM mods WHERE name = ?', (msg.user.name,)).fetchone():
        mod = True
    else:
        mod = False

    if ('@' + bot_name) in msg.text:
        if msg.user.name != 'streamElements' or 'soundAlerts' and msg.user.name not in engagement_list:
            engagement_list[msg.user.name] = {}
            engagement_list[msg.user.name]["convo"] = []
        await bot_command_handler(msg)
    elif msg.text.startswith('!points'):
        await points_command_handler(msg)
    elif msg.text.startswith('!addpoints') & mod:
        await add_command_handler(msg)
    elif msg.text.startswith('!removepoints') & mod:
        await remove_command_handler(msg)
    elif msg.text.startswith('!quote'):
        await quote_command_handler(msg)
    elif msg.text.startswith('!addquote') & mod:
        await addquote_command_handler(msg)
    elif msg.text.startswith('!removequote') & mod:
        await removequote_command_handler(msg)
    elif msg.text.startswith('!addmod') & mod:
        await addmod_command_handler(msg)
    elif msg.text.startswith('!removemod') & mod:
        await removemod_command_handler(msg)
    elif msg.text.startswith('!gamble'):
        await gamble_command_handler(msg)
    elif msg.text.startswith('!duel'):
        await duel_command_handler(msg)
    elif msg.text.startswith('!help'):
        await help_command_handler(msg)
    elif msg.user.name not in engagement_list:
        await engagement_handler(msg)
    
    if msg.user.name in engagement_list:
        # print(engagement_list)
        if 'messages' not in engagement_list[msg.user.name]:
            engagement_list[msg.user.name]["messages"] = 1
        else:
            # print(engagement_list[msg.user.name]["messages"])
            engagement_list[msg.user.name]["messages"] += 1
        
        if engagement_list[msg.user.name]["messages"] >= 5:
            engagement_list[msg.user.name]["messages"] = 0
            conn.execute('UPDATE users SET points = points + 1 WHERE name = ?', (msg.user.name,))
    
    conn.commit()
    conn.close()

async def engagement_handler(msg: ChatMessage):
    if engagement_mode:
        await msg.reply(f'hello @{msg.user.name}! I am a bot created by leisurellama, put @llamachop_bot in your message to talk to me! How are you doing today?')
        engagement_list[msg.user.name] = {}

async def help_command_handler(cmd: ChatCommand):
    cmd.reply(f'list of commands: !points, !gamble, !duel, !help')

# here we need to put in a function that will be executed when a user messages !bot in chat

# conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, user_message TEXT, author TEXT, timestamp TEXT, chatbot_response TEXT)')

async def bot_command_handler(cmd: ChatCommand, chat=None, convo=[]):
    if cmd is None:
        return
    trueMessage = cmd.text
    print(trueMessage)
    # print("below is cmd")
    # print(cmd.user['name'])
    response_object = ai.text_output(utterance=trueMessage, convo=convo)
    reply = response_object['response']

    # conn = sqlite3.connect('app.db')
    # conn.execute('INSERT INTO messages (user_message, author, timestamp, chatbot_response) VALUES (?, ?, ?, ?)', (trueMessage, cmd.user.name, datetime.datetime.now(), reply))
    # conn.commit()
    # conn.close()

    MAX_MESSAGE_LENGTH = 250

    if getattr(cmd, 'voice', None):
        if len(reply) > MAX_MESSAGE_LENGTH:
            half_length = len(reply) // 2
            first_half = reply[:half_length]
            second_half = reply[half_length:]
            await Chat.send_message(chat, room=TARGET_CHANNEL, text=first_half)
            await Chat.send_message(chat, room=TARGET_CHANNEL, text=second_half)
        else:
            await Chat.send_message(chat, room=TARGET_CHANNEL, text=reply)
        engagement_list[TARGET_CHANNEL]["convo"] = response_object['convo']
    else:
        if len(reply) > MAX_MESSAGE_LENGTH:
            half_length = len(reply) // 2
            first_half = reply[:half_length]
            second_half = reply[half_length:]
            await cmd.reply(f'{cmd.user.name}: {first_half}')
            await cmd.reply(f'{cmd.user.name}: {second_half}')
        else:
            await cmd.reply(f'{cmd.user.name}: {reply}')
        engagement_list[cmd.user.name]["convo"] = response_object['convo']
    
    if tts:
        # tts_proc = subprocess.Popen(['python', 'tts.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(reply)
        await text_to_speech.add_text(reply)

    if local_tts:
        media = vlc.MediaPlayer('output.mp3')
        media.play()


async def points_command_handler(cmd: ChatCommand):
    # this is going to access the database and return the number of points the user has
    conn = sqlite3.connect('app.db')
    reply = conn.execute('SELECT points FROM users WHERE name = ?', (cmd.user.name,)).fetchone()
    conn.close()
    await cmd.reply(f'{cmd.user.name} has {reply[0]} points')

async def add_command_handler(cmd: ChatCommand):
    # this is going to add the specified number of points to the specified user
    name = cmd.text.split(' ')[1]
    points = cmd.text.split(' ')[2]
    conn = sqlite3.connect('app.db')
    conn.execute('UPDATE users SET points = points + ? WHERE name = ?', (points, name)).fetchone()
    conn.commit()
    conn.close()

async def remove_command_handler(cmd: ChatCommand):
    # this is going to remove the specified number of points from the specified user
    name = cmd.text.split(' ')[1]
    points = cmd.text.split(' ')[2]
    conn = sqlite3.connect('app.db')
    conn.execute('UPDATE users SET points = points - ? WHERE name = ?', (points, name)).fetchone()
    conn.commit()
    conn.close()

async def quote_command_handler(cmd: ChatCommand):
    # this is going to return a random quote from the database
    conn = sqlite3.connect('app.db')
    res = conn.execute('SELECT id, quote FROM quotes ORDER BY RANDOM() LIMIT 1').fetchone()
    conn.close()
    # print(res)
    if res is not None:
        id, text = res
        await cmd.reply(f'#{id}: {text}')

async def addquote_command_handler(cmd: ChatCommand):
    # this is going to add the specified quote to the database
    quote = cmd.text[10:]
    conn = sqlite3.connect('app.db')
    conn.execute('INSERT INTO quotes (quote, author) VALUES (?, ?)', (quote, cmd.user.name))
    await cmd.reply(f'Added quote: {quote}')
    conn.commit()
    conn.close()

async def removequote_command_handler(cmd: ChatCommand):
    # this is going to remove the specified quote from the database
    quote = cmd.text.split(' ')[1]
    conn = sqlite3.connect('app.db')
    conn.execute('DELETE FROM quotes WHERE id = ?', (quote,))
    cmd.reply(f'Removed quote #{quote}')
    conn.commit()
    conn.close()

async def addmod_command_handler(cmd: ChatCommand):
    # this is going to add the specified user to the mods table
    name = cmd.text.split(' ')[1]
    conn = sqlite3.connect('app.db')
    conn.execute('INSERT INTO mods (name) VALUES (?)', (name,))
    await cmd.reply(f'{name} has been added to the mods list')
    conn.commit()
    conn.close()

async def removemod_command_handler(cmd: ChatCommand):
    # this is going to remove the specified user from the mods table
    name = cmd.text.split(' ')[1]
    conn = sqlite3.connect('app.db')
    if name != TARGET_CHANNEL:
        conn.execute('DELETE FROM mods WHERE name = ?', (name,))
        await cmd.reply(f'{name} has been removed from the mods list')
    else:
        await cmd.reply(f'{name} cannot be removed from the mods list')
    conn.commit()
    conn.close()

async def gamble_command_handler(cmd: ChatCommand):
    # this is going to allow the user to gamble their points
    # for this to work we need a function that will return a random number between 1 and 2.
    random_number = random.randint(1, 2)
    points = cmd.text.split(' ')[1]
    conn = sqlite3.connect('app.db')
    points_total = conn.execute('SELECT points FROM users WHERE name = ?', (cmd.user.name,)).fetchone()
    if int(points) > int(points_total[0]):
        await cmd.reply(f'{cmd.user.name} does not have enough points to gamble')
    elif random_number == 1:
        conn.execute('UPDATE users SET points = points + ? WHERE name = ?', (points, cmd.user.name)).fetchone()
        await cmd.reply(f'{cmd.user.name} has won {points} points')
    else:
        conn.execute('UPDATE users SET points = points - ? WHERE name = ?', (points, cmd.user.name)).fetchone()
        await cmd.reply(f'{cmd.user.name} has lost {points} points')
    conn.commit()
    conn.close()

async def duel_command_handler(cmd: ChatCommand):
    # this is going to allow the user to duel another user
    # for this we are also going to need a function that returns a random number between 1 and 2.
    random_number = random.randint(1, 2)
    opponent = cmd.text.split(' ')[1]
    points = cmd.text.split(' ')[2]
    conn = sqlite3.connect('app.db')
    points_total = conn.execute('SELECT points FROM users WHERE name = ?', (cmd.user.name,)).fetchone()
    opponent_points_total = conn.execute('SELECT points FROM users WHERE name = ?', (opponent,)).fetchone()
    if int(points) > int(points_total[0]):
        await cmd.reply(f'{cmd.user.name} does not have enough points to duel')
    elif int(points) > int(opponent_points_total[0]):
        await cmd.reply(f'{opponent} does not have enough points to duel')
    elif random_number == 1:
        conn.execute('UPDATE users SET points = points + ? WHERE name = ?', (points, cmd.user.name)).fetchone()
        conn.execute('UPDATE users SET points = points - ? WHERE name = ?', (points, opponent)).fetchone()
        await cmd.reply(f'{cmd.user.name} has beat {opponent}! they won {points} points')
    else:
        conn.execute('UPDATE users SET points = points - ? WHERE name = ?', (points, cmd.user.name)).fetchone()
        conn.execute('UPDATE users SET points = points + ? WHERE name = ?', (points, opponent)).fetchone()
        await cmd.reply(f'{cmd.user.name} has lost to {opponent}! they lost {points} points')
    conn.commit()
    conn.close()

async def recording_handler(chat):
    if TARGET_CHANNEL in engagement_list and 'convo' in engagement_list[TARGET_CHANNEL]:
        convo = engagement_list[TARGET_CHANNEL]['convo']
    elif TARGET_CHANNEL in engagement_list and 'convo' not in engagement_list[TARGET_CHANNEL]:
        engagement_list[TARGET_CHANNEL]['convo'] = []
        convo = engagement_list[TARGET_CHANNEL]['convo']
    else:
        engagement_list[TARGET_CHANNEL] = {}
        engagement_list[TARGET_CHANNEL]['convo'] = []
        convo = engagement_list[TARGET_CHANNEL]['convo']
    
    if stt:
        response = await start_listen_thread()
        if response is not None:
            response['user'] = {}
            response['user']['name'] = TARGET_CHANNEL

        if response:
            await bot_command_handler(response, chat, convo)
        else:
            print('No response')
        # stt_proc = subprocess.Popen(['python', 'stt.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # text = stt_proc.stdout.read()
        # print(text)
        # await bot_command_handler(text, chat, convo)
    


# set up the connection to Twitch
async def twitch_connect():
    # info pulled from the .env file for twitch authentication
    twitch = await Twitch(CLIENT_ID, CLIENT_SECRET)
    if user_credentials is not None:
        token = user_credentials['token']
        refresh_token = user_credentials['refresh_token']
        # if login fails, the token will refresh and save
        try:
            await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
        except:
            auth = UserAuthenticator(twitch, USER_SCOPE, force_verify=False)
            token, refresh_token = await auth.authenticate()
            await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
            with open('user_credentials.json', 'w') as f:
                json.dump({'token': token, 'refresh_token': refresh_token}, f)
        print("User credentials loaded")
    else: # if there is no saved user credential, it will authenticate and save
        auth = UserAuthenticator(twitch, USER_SCOPE, force_verify=False)
        token, refresh_token = await auth.authenticate()
        await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
        with open('user_credentials.json', 'w') as f:
            json.dump({'token': token, 'refresh_token': refresh_token}, f)
            print("User credentials saved")

    
    print("Twitch connection established")

    chat = await Chat(twitch)
    chat.register_event(ChatEvent.READY, on_ready)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()

    try:
        # input("Press enter to stop the bot...\n")
        print(f"waiting for {push_to_talk} to be pressed")
        while True:
            keyboard.wait(push_to_talk)
            await recording_handler(chat)
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("keyboard interrupt")
        pass
    finally:
        print("Stopping bot")
        chat.stop()
        await twitch.close()

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8080, debug=True)
asyncio.run(twitch_connect())
