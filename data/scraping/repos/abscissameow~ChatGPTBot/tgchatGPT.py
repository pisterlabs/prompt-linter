import openai
from telegram import Update, Bot
from telegram.ext import CallbackContext, CommandHandler, Updater, MessageHandler, Filters
import sys, os
import json
from copy import deepcopy
from gtts import gTTS
from pydub import AudioSegment

DEBUG = False

# insert corresponding tokens in terminal like that: python3 tgchatGPT.py token1 token2
openai.api_key, TOKEN = sys.argv[1:3]

# create/fill data
DIR_path    = os.path.dirname(os.path.abspath(__file__))
DATA_path   = DIR_path + "/data"
IDS_path    = DATA_path + "/IDS.json"
MEMORY_path = DATA_path + "/MEMORY.json"
if not os.path.exists(DATA_path): os.makedirs(DATA_path)
for item in os.listdir(DATA_path):
  if item not in IDS_path: os.remove(DATA_path + '/' + item)
if not os.path.exists(IDS_path): 
  with open(IDS_path,    'w+') as f: json.dump({}, f)
with open(IDS_path,    'r') as f: IDS    = json.load(f)
if not os.path.exists(MEMORY_path): 
  with open(MEMORY_path, 'w+') as f: json.dump({}, f)
with open(MEMORY_path, 'r') as f: MEMORY = json.load(f)

# CONSTANTS
DEFAULT_DICT    = {'state':'chat', 'a':list(), 'q':list()}
MEMORY_REQUESTS = 7
MAX_TOKENS = 2800
MYID, HERID = 283460642, 284672038
GODS = {MYID : "おはよう　おもさま", HERID : "おはよう むすび　ちゃん"}
  
# GPT chat api implementation
def GPTchat(prompt):
  return openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt},
          ],)['choices'][0]['message']['content']

# GPT img api implementation
def GPTimg(prompt):
  return openai.Image.create(
    prompt=prompt,
    n=1,
    size="1024x1024")['data'][0]['url']

# save user's data to memory
def _fill(chat_id, username):
  if username not in IDS:
    IDS[str(username)] = chat_id
    IDS[str(chat_id) ] = str(username)
    with open(IDS_path, 'w+') as fp: json.dump(IDS, fp)
  if chat_id not in MEMORY: MEMORY[chat_id] = deepcopy(DEFAULT_DICT)

# /start command
def start_command(update: Update, context: CallbackContext) -> None:
  chat_id  = update.message.chat_id
  _fill(chat_id, update.message.from_user.username)
  if chat_id in GODS: update.message.reply_text(GODS[chat_id])
  else:               update.message.reply_text('おはよう　ともだち!')

# /img command
def img(update: Update, context: CallbackContext) -> None:
  chat_id  = update.message.chat_id 
  _fill(chat_id, update.message.from_user.username)
  MEMORY[chat_id]['state'] = 'img'
  update.message.reply_text('рисоватб: ON')

# /chat command
def chat(update: Update, context: CallbackContext) -> None:
  chat_id = update.message.chat_id
  _fill(chat_id, update.message.from_user.username)
  MEMORY[chat_id]['state'] = 'chat'
  update.message.reply_text('чатб: ON')


def _cap_memory(id):
  T = len(''.join(MEMORY[id]['q']+MEMORY[id]['a']).split())
  if len(MEMORY[id]['q']) >= MEMORY_REQUESTS or T > MAX_TOKENS:
    MEMORY[id]['a'].pop(0)
    MEMORY[id]['q'].pop(0)
    _cap_memory(id)

def _handle_memory_chat(chat_id, text):
  _cap_memory(chat_id)
  prompt = '\n'.join([i for j in zip(MEMORY[chat_id]['q'],MEMORY[chat_id]['a'])\
                        for i in j]) + '\n' + text
  answer = GPTchat(prompt)
  MEMORY[chat_id]['q'].append(text)
  MEMORY[chat_id]['a'].append(answer)
  return answer

# handler for any text
def handleGPT(update: Update, context: CallbackContext):
  try:
    chat_id = update.message.chat_id
    msg     = update.message.text.lower()
    _fill(chat_id, update.message.from_user.username)

    # using GPT image api
    if MEMORY[chat_id]['state'] == 'img':
      # if first token of msg is integer then make this number of replies:
      n, text = msg.split(' ', 1)
      if n.isdigit():
        for i in range(int(n)):
          update.message.reply_text(f"{i+1}/{n}\n"+GPTimg(text))
      else: update.message.reply_text(GPTimg(msg))
      
    # using GPT chat api
    else:
      update.message.reply_text(_handle_memory_chat(chat_id, msg))
  except Exception as e:
    update.message.reply_text('я сломалосб:\n' + str(e))

# handle voice files
def handleAudio(update: Update, context: CallbackContext):
  try:
    chat_id = update.message.chat_id
    _fill(chat_id, update.message.from_user.username)

    temp_path = DATA_path + f'/{chat_id}.mp3'
    with open(temp_path, 'w+'): pass # create temp file

    update.message.voice.get_file().download(temp_path) # get .ogg voice file
    AudioSegment.from_ogg(temp_path).export(temp_path, format="mp3") # convert .ogg to .mp3

    with open(temp_path, 'rb') as fp: 
      prompt = openai.Audio.translate("whisper-1", fp)['text'] # response

    # reply with text/pic/voice depending on first_word:
    first_word = prompt.split()[0].lower()

    if first_word in ['write', 'print', 'type']: # txt
      update.message.reply_text(_handle_memory_chat(chat_id, prompt+"\nОтветь на русском языке"))

    elif first_word in ['draw', 'paint', 'picture']: # pic
      update.message.reply_text(GPTimg(prompt))

    else: # voice
      gTTS(_handle_memory_chat(chat_id, prompt+"\nОтветь на русском языке"),
           lang='ru', slow=False, lang_check=False).save(temp_path)    # generate voice response
      with open(temp_path, 'rb') as fp: update.message.reply_voice(fp) # reply voice
      os.remove(temp_path) # remove temp file
  except Exception as e:
    update.message.reply_text('я сломалосб:\n' + str(e))

# clear user memory:
def clear(update: Update, context: CallbackContext) -> None:
  try:
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    _fill(chat_id, username)
    MEMORY.pop(chat_id)
    update.message.reply_text('всё чисто')
  except Exception as e:
    update.message.reply_text('я сломалосб:\n' + str(e))

# help command
def help(update: Update, context: CallbackContext):
  chat_id = update.message.chat_id
  username = update.message.from_user.username
  _fill(chat_id, username)
  Help = '''\
1. /chat - режим чата
2. /pic - режим картинок
3. /clear - очищает память для режима чата
4. аудиосообщения:
  a) если первое слово в аудио "нарисуй", то оно нарисует
  b) если первое слово в аудио "напиши", то ответит текстом
  c) иначе: пришлет сгенерированное аудиосообщение
  
больше help:  
  https://github.com/abscissameow/ChatGPTBot
  '''
  update.message.reply_text(Help)

# debug staff
def _void(update: Update, context: CallbackContext) -> None: # erase data
  if update.message.chat_id in GODS:
    global MEMORY
    MEMORY = {}
    for item in os.listdir(DATA_path):
      if item not in IDS_path: os.remove(DATA_path + '/' + item)
    update.message.reply_text('しんでいる')
  else:
    _fill(update.message.chat_id, update.message.from_user.username)
    update.message.reply_text('NOT ALLOWED')

def _get(update: Update, context: CallbackContext) -> None: 
  if update.message.chat_id in GODS:
    if DEBUG:
      users = '\n'.join([IDS[str(key)]+' : '+str(len(MEMORY[key]['q'])) for key in MEMORY])
      if not users:
        update.message.reply_text('empty')
      else:
        with open(MEMORY_path, 'w+') as f: json.dump(MEMORY, f)
        update.message.reply_text(users)
        update.message.reply_text("\n\n".join(
          [f"{n+1}) {IDS[str(key)]}\n\
          {(MEMORY[key]['q'][-1] if MEMORY[key]['q'] else None)}"[:4096//len(MEMORY)]\
          for n,key in enumerate(MEMORY)]))
  else:
    update.message.reply_text('NOT ALLOWED')

def _send(update: Update, context: CallbackContext) -> None: # send message
  if update.message.chat_id in GODS:
    if DEBUG:
      _, username, msg = update.message.text.split(' ', 2)
      Bot(token=TOKEN).send_message(chat_id = IDS[username], text = msg)
  else:
    update.message.reply_text('NOT ALLOWED')

# put all together and start pooling
handlers = [
  CommandHandler('start', start_command),
  CommandHandler('chat',  chat),
  CommandHandler('img',   img),
  CommandHandler('clear', clear),
  CommandHandler('help', help),
  CommandHandler('void', _void),
  CommandHandler('get',  _get),
  CommandHandler('send', _send),
  MessageHandler(Filters.text,  handleGPT),
  MessageHandler(Filters.voice, handleAudio),
]
updater = Updater(TOKEN, workers=100)
for handler in handlers:
  updater.dispatcher.add_handler(handler)
updater.start_polling()
print('Start bot')
updater.idle()