from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import json, os, string, sys, threading, logging, time, re, random
import discord
import openai

##########
#Settings#
##########

#OpenAI API key
aienv = os.getenv('OPENAI_KEY')
if aienv == None:
    openai.api_key = "YOUR OPENAI API KEY GOES HERE"
else:
    openai.api_key = aienv
print(aienv)

#Discord bot key
denv = os.getenv('DISCORD_KEY')
if denv == None:
    dkey = "YOUR DISCORD BOT KEY GOES HERE"
else:
    dkey = denv
print(denv)

# Lots of console output
debug = True

#Defaults
user = 'Human'
botname = 'AI'
cache = None
qcache = None
chat_log = None
running = False
# Max chat log length (A token is about 4 letters and max tokens is 2048)
max = int(3000)

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

completion = openai.Completion()

##################
#Command handlers#
##################

def retry(message, username, botname):
    """Send a message when the command /retry is issued."""
    new = True
    rep = interact(message, username, botname, new)
    return rep

################
#Main functions#
################

def limit(text, max):
    if (len(text) >= max):
        inv = max * -1
        print("Reducing length of chat history... This can be a bit buggy.")
        nl = text[inv:]
        text = re.search(r'(?<=\n)[\s\S]*', nl).group(0)
        return text
    else:
        return text

def run(message, username, botname):
    new = False
    rep = interact(message, username, botname, new)
    return rep

def ask(username, botname, question, chat_log=None):
    if chat_log is None:
        chat_log = 'The following is a roleplay between two users:\n\n'
    now = datetime.now()
    ampm = now.strftime("%I:%M %p")
    t = '[' + ampm + '] '
    prompt = f'{chat_log}{t}{username}: {question}\n{t}{botname}:'
    response = completion.create(
        prompt=prompt, engine="davinci", stop=['\n'], temperature=0.9,
        top_p=1, frequency_penalty=15, presence_penalty=2, best_of=3,
        max_tokens=250)
    answer = response.choices[0].text.strip()
    return answer

def append_interaction_to_chat_log(username, botname, question, answer, chat_log=None):
    if chat_log is None:
        chat_log = 'The following is a roleplay between two users:\n\n'
    chat_log = limit(chat_log, max)
    now = datetime.now()
    ampm = now.strftime("%I:%M %p")
    t = '[' + ampm + '] '
    return f'{chat_log}{t}{username}: {question}\n{t}{botname}: {answer}\n'
	
def interact(message, username, botname, new):
    global chat_log
    global cache
    global qcache
    print("==========START==========")
    text = str(message)
    analyzer = SentimentIntensityAnalyzer()
    if new != True:
        vs = analyzer.polarity_scores(text)
        if debug == True:
            print("Sentiment of input:\n")
            print(vs)
        if vs['neg'] > 1:
            rep = 'Input text is not positive. Input text must be of positive sentiment/emotion.'
            return rep
    if new == True:
        if debug == True:
            print("Chat_LOG Cache is...")
            print(cache)
            print("Question Cache is...")
            print(qcache)
        chat_log = cache
        question = qcache
    if new != True:
        question = text
        qcache = question
        cache = chat_log
    try:
        print('TEST')
        answer = ask(username, botname, question, chat_log)
        print('TEST')
        if debug == True:
            print("Input:\n" + question)
            print("Output:\n" + answer)
            print("====================")
        stripes = answer.encode(encoding=sys.stdout.encoding,errors='ignore')
        decoded	= stripes.decode("utf-8")
        out = str(decoded)
        vs = analyzer.polarity_scores(out)
        if debug == True:
            print("Sentiment of output:\n")
            print(vs)
        if vs['neg'] > 1:
            rep = 'Output text is not positive. Censoring. Use /retry to get positive output.'
            return rep
        chat_log = append_interaction_to_chat_log(username, botname, question, answer, chat_log)
        print(chat_log)
        return out

    except Exception as e:
        print(e)
        errstr = str(e)
        return errstr

#####################
# End main functions#
#####################

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        global running
        global botname
        global username
        global chat_log
        global cache
        global qcache
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return
        if message.content.startswith('!start'):
            user = 'Human'
            botname = 'AI'
            chat_log = None
            cache = None
            qcache = None
            running = True
            await message.reply('You have started the bot. Commands are !start, !stop, !botname (name of your desired rp partner), !username (your rp character) and !rp (text)', mention_author=False)
        if message.content.startswith('!stop'):
            user = 'Human'
            botname = 'AI'
            chat_log = None
            cache = None
            qcache = None
            running = False
            await message.reply('You have stopped the bot.', mention_author=False)
        if message.content.startswith('!reset'):
            username = 'Human'
            botname = 'AI'
            chat_log = None
            cache = None
            qcache = None
            await message.reply('You have reset the bot.', mention_author=False)
        if message.content.startswith('!botname'):
            botname = re.search(r'(?<=!botname ).*[^.]*', message.content)
            name = botname.group(0)
            botname = str(name)
            reply = 'Bot character set to: ' + botname
            await message.reply(reply, mention_author=False)
        if message.content.startswith('!username'):
            username = re.search(r'(?<=!username ).*[^.]*', message.content)
            name = username.group(0)
            username = str(name)
            reply = 'Your character set to: ' + username
            await message.reply(reply, mention_author=False)
        if message.content and running == True:
            if message.content.startswith('!retry'):
                conts = 'null'
                rep = retry(conts, username, botname)
                await message.reply(rep, mention_author=False)
            if message.content.startswith('!rp'):
                content = re.search(r'(?<=!rp ).*[^.]*', message.content)
                cont = content.group(0)
                conts = str(cont)
                rep = run(conts, username, botname)
                await message.reply(rep, mention_author=False)

if __name__ == '__main__':
    client = MyClient()
    client.run(dkey)
