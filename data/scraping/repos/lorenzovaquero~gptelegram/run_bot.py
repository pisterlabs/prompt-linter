#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simple Bot to reply to Telegram messages using OpenAI
"""
This Bot uses the Updater class to handle the bot.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import argparse
import yaml
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Chat
import logging, os, random, sys, datetime, glob, time
from dateutil.relativedelta import relativedelta
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential  # for exponential backoff
import subprocess
import pickle
import shutil
import copy

CURRENT_DIR = os.path.dirname(__file__)

STORE_CONV = False
LOGS_FOLDER = 'log'
CHAT_FOLDER = 'chats'
PICS_FOLDER = 'pic'
CHAT_PKL = 'chat.pkl'

DEFAULT_TEXT_ENGINE = 'text-davinci-003'  # 'text-davinci-002' | 'text-davinci-003'
MAX_OUT_TOKENS = 250
MAX_IN_CHARS = 500
MAX_CONTEXT_MESSAGES = 6

STABLE_DIFFUSION_PATH = ""  # Add stable-diffusion path

DEFAULT_HUMAN = "Lorenzo"
DEFAULT_BOT = "Dabotni"

DEFAULT_PROMPT = """The following is a conversation between {bot} and his friend {human}. {bot} always does what {human} tells him to do."""

PROMPT_LINK = """

{human}: {__MSG__}

{bot}:"""

CURRENT_PROMPT = None
CHAT_WHITELIST = None


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)



class SafeDict(dict):  # To not replace stuff if it's not available
    def __missing__(self, key):
        return '{' + key + '}'


def whitelist_filter(func):
    def inner(*args, **kwargs):
        
        if len(args) < 2:
            return
        update = args[0]
        context = args[1]
        
        if update is None or context is None:
            return
        
        interaction_allowed = check_interaction_allowed(whitelist=CHAT_WHITELIST, update=update)
        
        human_name = get_human_name(from_user=update.message.from_user)
        if human_name is None:
            human_name = DEFAULT_HUMAN
        
        user = update.message.from_user
        chat_id = update.effective_chat.id
        chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
        telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
        
        if not interaction_allowed:
            if STRICT_WHITE_LIST:  # If the white-list is strict, we reply with an error and return
                
                logger.warning('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) has been strictly whitelisted. Sends ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id, user_msg=update.message.text))
                context.bot.send_message(chat_id=chat_id, text='An error has occured. Please try again later.\nIf the problem persists, please <a href="https://latlmes.com/chatbot/help">contact support</a>.', disable_web_page_preview=True, parse_mode='HTML')
                return
            
            else:
                  # If the white-list is strict, we reply with the default and safe PROMPT
                  logger.warning('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) has been shadow whitelisted.'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username))
                  
                  return func(*args, **kwargs, prompt_text=DEFAULT_PROMPT)
            
        else:
            return func(*args, **kwargs, prompt_text=CURRENT_PROMPT)  # If the user is whitelisted, we load the "CURRENT_PROMPT", a.k.a. the prompt in the file
    
    return inner


def check_interaction_allowed(whitelist, update):
    if whitelist is None:
        return True  # If there is no whitelist, we allow everything
    
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    
    if chat_id in whitelist['CHATS']:
        return True
    
    if user_id in whitelist['USERS']:
        return True
    
    return False

def check_admin_allowed(whitelist, update):
    if whitelist is None:
        return False  # If there is no whitelist, we block everything
    
    user_id = update.message.from_user.id
    
    if user_id in whitelist['ADMINS']:
        return True
    
    return False


def convert_tokens_to_dollars(num_tokens):
    if DEFAULT_TEXT_ENGINE.startswith('text-davinci'):
        multiplier = 0.00002  # $0.0200 / 1K tokens
    
    else:
        raise NotImplementedError
    
    dollars = num_tokens * multiplier
    
    return dollars

def get_human_name(from_user):
    if hasattr(from_user, 'first_name') and hasattr(from_user, 'last_name'):
        if from_user['first_name'] is not None and len(from_user['first_name']) > 0:
            if from_user['last_name'] is not None and len(from_user['last_name']) > 0:
                return '{} {}'.format(from_user['first_name'].strip(), from_user['last_name'].strip())
    
    for field in ['first_name', 'last_name', 'username']:  # By order of preference
        if hasattr(from_user, field) and from_user[field] is not None and len(from_user[field]) > 0:
            return from_user[field].strip()
    
    return None  # If there is no luck


def get_chat_folder(effective_chat, reset_chat=False, hard_reset=False):
    """Gets (or creates) a chat folder. Reset chat deletes the conversation data, not the metadata (unless hard_reset is set to True)"""
    chat_id = str(effective_chat.id)
    chat_folder = os.path.join(CURRENT_DIR, CHAT_FOLDER, chat_id)
    
    if reset_chat and os.path.isdir(chat_folder):
        if hard_reset:  # We erase all conversation/pictures/whatever
            shutil.rmtree(chat_folder)
        
        else:  # We erase conversation history and pictures folder
            # TODO: Delete pictures folder
            
            pkl_file = os.path.join(chat_folder, CHAT_PKL)
            with open(pkl_file, "rb") as chat_file:
                chat_data = pickle.load(chat_file)
            
            chat_data['chat'] = []  # Reset conv history
            chat_data['metadata'] = effective_chat.to_dict()  # Also update metadata
            with open(pkl_file, "wb") as chat_file:
                pickle.dump(chat_data, chat_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not os.path.isdir(chat_folder):
        os.makedirs(chat_folder, exist_ok=True)
        os.makedirs(os.path.join(chat_folder, PICS_FOLDER), exist_ok=True)
        pkl_file = os.path.join(chat_folder, CHAT_PKL)
        
        with open(pkl_file, "wb") as chat_file:
            pickle.dump({'metadata': effective_chat.to_dict(), 'chat': [], 'members': {}, 'total_tokens': 0, 'usage': []}, chat_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return chat_folder


def save_interaction(chat_folder, user_name, msg_text):
    pkl_file = os.path.join(chat_folder, CHAT_PKL)
    with open(pkl_file, "rb") as chat_file:
        chat_data = pickle.load(chat_file)
    
    with open(pkl_file, "wb") as chat_file:
        new_data = {'user': user_name, 'msg': msg_text}
        chat_data['chat'].append(new_data)
        # TODO: Maybe limit it so we delete oldest interactions?

        pickle.dump(chat_data, chat_file, protocol=pickle.HIGHEST_PROTOCOL)

def save_usage(chat_folder, user, num_tokens):
    user_id = user.id
    
    pkl_file = os.path.join(chat_folder, CHAT_PKL)
    with open(pkl_file, "rb") as chat_file:
        chat_data = pickle.load(chat_file)
    
    chat_data['total_tokens'] += num_tokens
    chat_data['usage'].append((time.time(), user_id, num_tokens))  # triplets [time, user, tokens]
    
    
    if user_id not in chat_data['members']:
        chat_data['members'][user_id] = {'metadata': user.to_dict(), 'total_tokens': 0}
    
    chat_data['members'][user_id]['total_tokens'] += num_tokens
    
    with open(pkl_file, "wb") as chat_file:
        pickle.dump(chat_data, chat_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_interaction(chat_folder):
    pkl_file = os.path.join(chat_folder, CHAT_PKL)
    
    with open(pkl_file, "rb") as pkl_file:
        chat_data = pickle.load(pkl_file)
    
    return chat_data


def clean_query(msg_text, max_chars=MAX_IN_CHARS):
    initial_len = len(msg_text)
    if msg_text.startswith('/'):
        msg_text = msg_text.split(' ', 1)[-1]  # We remove the '/AI ' (or whatever) part
        
        if len(msg_text) == initial_len:
            # This implies that the command was only the '/AI'
            msg_text = ''
            
        msg_text = msg_text.strip()
    
    if len(msg_text) == 0:
        # raise ValueError('Empty query')
        return msg_text
    
    msg_text = msg_text[:max_chars]  # So we don't end up poor

    # We add a trailing dot if the text does not end with one
    last_char = msg_text[-1]
    if last_char not in set(['.', '!', '?', ':']):
        msg_text = msg_text + '.'
    
    return msg_text


def clean_answer(msg_text):
    answ = msg_text.strip()
    
    return answ


def assemble_context(chat_data, max_messages=MAX_CONTEXT_MESSAGES, chatcompletion_format=False):
    messages_list = chat_data['chat']
    
    if len(messages_list) == 0:
        return None
    
    messages_list = messages_list[-max_messages:]  # We limit the length
    
    if chatcompletion_format:
        context = []
        for message in messages_list:
            if message['user'] == DEFAULT_BOT:
                role = "assistant"
            else:
                role = "user"
            context.append({"role": role, "content": message['msg']})
    
    else:
        context = ""
        for i, message in enumerate(messages_list):
            new_msg = '{}: {}'.format(message['user'], message['msg'])
            
            if i < len(messages_list) - 1:
                new_msg = new_msg + '\n\n'
            
            context = context + new_msg
    
    return context


def assemble_prompt(prompt_text=DEFAULT_PROMPT, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT, context=None, chatcompletion_format=False):
    prompt = prompt_text.format_map(SafeDict(human=human_name, bot=bot_name))
    
    if chatcompletion_format:
        prompt = [{"role": "system", "content": prompt}]
        if context is not None and len(context) > 0:
            prompt.extend(context)
        
    else:
        if context is not None and len(context) > 0:
            prompt = prompt + '\n\n{__CONTEXT__}'.format_map(SafeDict(__CONTEXT__=context))
        
        prompt = prompt + PROMPT_LINK.format_map(SafeDict(human=human_name, bot=bot_name))

    return prompt


def assemble_openai_query(prompt, query, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT, chatcompletion_format=False):
    if chatcompletion_format:
        query = [{"role": "user", "content": query}]
        
        if prompt is not None and len(prompt) > 0:
            prompt = copy.deepcopy(prompt)
            prompt.extend(query)
            query = prompt
        
    else:
        query = prompt.format_map(SafeDict(__MSG__=query))  # TODO: I should escape the other potential braces in the conversation
    
    return query


def generate_image(prompt_text):
    # TODO: Do in a better way
    prompt_file = os.path.abspath(os.path.join(CURRENT_DIR, "pic.txt"))
    output_dir = os.path.abspath(os.path.join(CURRENT_DIR, "bot_imgs"))
    with open(prompt_file, "w") as f:
        f.write(prompt_text)
    
    # Generate img
    generator_script = os.path.abspath(os.path.join(STABLE_DIFFUSION_PATH, "scripts", "txt2img.py"))
    config_file = os.path.abspath(os.path.join(STABLE_DIFFUSION_PATH, "configs", "stable-diffusion", "v1-inference.yaml"))
    ckpt_file = os.path.abspath(os.path.join(STABLE_DIFFUSION_PATH, "models", "ldm", "stable-diffusion-v1", "model.ckpt"))
    subprocess.call(["python", generator_script,
                     "--config", config_file,
                     "--ckpt", ckpt_file,
                     "--prompt-file", prompt_file,
                     "--outdir", output_dir,
                     "--H", "640",
                     "--W", "576",
                     "--seed", "42",
                     "--ddim_steps", "50",
                     "--n_samples", "1",
                     "--n_iter", "1",
                     "--skip_grid"])
    
    # Retrieve img
    list_of_files = glob.glob(os.path.join(output_dir, 'samples', '*')) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    
    return latest_file


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_openai_answer_completion(msg_text, text_engine=DEFAULT_TEXT_ENGINE, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT, user_id='', max_tokens=MAX_OUT_TOKENS):
    start_sequence = "\n{}:".format(bot_name)
    restart_sequence = "\n{}:".format(human_name)

    response = completion_with_backoff(
      engine=text_engine,
      prompt=msg_text,
      temperature=0.9,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=[restart_sequence, start_sequence],
      user=str(user_id)
    )
    answ = response.choices[0].text
    total_tokens = response.usage.total_tokens

    return answ, total_tokens

def get_openai_answer_chat(msg_text, text_engine=DEFAULT_TEXT_ENGINE, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT, user_id='', max_tokens=MAX_OUT_TOKENS):
    start_sequence = "\n{}:".format(bot_name)
    restart_sequence = "\n{}:".format(human_name)

    response = chat_with_backoff(
      model="gpt-4",
      messages=msg_text,
      temperature=0.9,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=[restart_sequence, start_sequence],
      user=str(user_id)
    )
    answ = response.choices[0].message
    answ = answ.content
    total_tokens = response.usage.total_tokens

    return answ, total_tokens

def talk_to_openai(message, update, store_conv=False, prompt_text=DEFAULT_PROMPT, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT, chatcompletion_format=True):
    user = update.message.from_user
    chat_id = update.effective_chat.id
    
    # If is admin, we will allow more tokens
    is_admin = check_admin_allowed(whitelist=CHAT_WHITELIST, update=update)
    
    # We create (or get) file with chat_id conversation
    chat_folder = get_chat_folder(effective_chat=update.effective_chat)
	
    if store_conv:
        # Retrieve conv_context
        conv_context_data = load_interaction(chat_folder=chat_folder)
    
        # Store question in file
        save_interaction(chat_folder=chat_folder, user_name=human_name, msg_text=message)
        
        context_txt = assemble_context(chat_data=conv_context_data, chatcompletion_format=chatcompletion_format)
    
    else:
        context_txt = None
    
    prompt = assemble_prompt(prompt_text=prompt_text, human_name=human_name,
	                         bot_name=bot_name, context=context_txt,
	                         chatcompletion_format=chatcompletion_format)
	
    openai_query = assemble_openai_query(prompt=prompt, query=message, chatcompletion_format=chatcompletion_format)
    
    num_chars = [len(q['content']) for q in openai_query]
    max_tokens = int(8192 - sum(num_chars) / 3.5) if is_admin else MAX_OUT_TOKENS
    
    answ, total_tokens = get_openai_answer_chat(openai_query, human_name=human_name, bot_name=bot_name, user_id=update.message.from_user.id,
	                                            max_tokens=max_tokens)
    
    answ = clean_answer(answ)
    
    save_usage(chat_folder=chat_folder, user=user, num_tokens=total_tokens)
    
    if store_conv:
        # Store answer in file
        save_interaction(chat_folder=chat_folder, user_name=bot_name, msg_text=answ)
	
    return answ


@whitelist_filter
def bot_pic_handler(update, context, prompt_text=None):
    if prompt_text is None:
        prompt_text = CURRENT_PROMPT
    
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    
    msg = clean_query(update.message.text)
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) sends /PIC ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id, user_msg=msg))
    
    if len(msg) == 0:
        return
    
    answ = talk_to_openai(message=msg, update=update, store_conv=STORE_CONV, prompt_text=prompt_text, human_name=human_name, bot_name=DEFAULT_BOT)
    
    if len(answ) == 0:
        return
    
    logger.info('Chat {chat_id} ({chat_name}) - BOT ({bot_name}) creates answer for /PIC ({message_id}): "{bot_answ}"'.format(chat_name=chat_name, chat_id=chat_id, bot_name=DEFAULT_BOT, message_id=update.message.message_id, bot_answ=answ))
    
    context.bot.send_message(chat_id=update.effective_chat.id, text="Let me think...")
    
    # TODO: Implement better!
    latest_file = generate_image(prompt_text=answ)
    
    logger.info('Chat {chat_id} ({chat_name}) - BOT ({bot_name}) answers /PIC ({message_id}, {file_name}): "{bot_answ}"'.format(chat_name=chat_name, chat_id=chat_id, bot_name=DEFAULT_BOT, message_id=update.message.message_id, file_name=latest_file, bot_answ=answ))
    
    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo=open(latest_file, 'rb'),
                           caption=answ,
                           reply_to_message_id=update.message.message_id,
                           allow_sending_without_reply=True)


@whitelist_filter
def bot_ai_handler(update, context, prompt_text=None):
    if prompt_text is None:
        prompt_text = CURRENT_PROMPT
    
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    
    msg = clean_query(update.message.text)
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) sends /AI ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id, user_msg=msg))
    
    if len(msg) == 0:
        return
    
    answ = talk_to_openai(message=msg, update=update, store_conv=STORE_CONV, prompt_text=prompt_text, human_name=human_name, bot_name=DEFAULT_BOT)
    
    logger.info('Chat {chat_id} ({chat_name}) - BOT ({bot_name}) answers /AI ({message_id}): "{bot_answ}"'.format(chat_name=chat_name, chat_id=chat_id, bot_name=DEFAULT_BOT, message_id=update.message.message_id, bot_answ=answ))
    
    if len(answ) == 0:
        return
	
    context.bot.send_message(chat_id=chat_id, text=answ)


def bot_TEXT_handler(update, context):
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    
    msg = clean_query(update.message.text)
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) sends TEXT ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id, user_msg=msg))
	
	# We create (or get) file with chat_id conversation
    chat_folder = get_chat_folder(effective_chat=update.effective_chat)
	
    if STORE_CONV:
        # Store question in file
        save_interaction(chat_folder=chat_folder, user_name=human_name, msg_text=msg)


def bot_reset_handler(update, context):  # Resets the conversation memory of the chat
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) sends /RESET ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id, user_msg=update.message.text))
    
    chat_folder = get_chat_folder(effective_chat=update.effective_chat, reset_chat=True)
    
    logger.info('Chat {chat_id} ({chat_name}) - Deleted chat for user {user_id} ({user_name}, {t_username}): "{chat_folder}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, chat_folder=chat_folder))
    
    context.bot.send_message(chat_id=chat_id, text="I forgor 💀")


def get_registered_chats(context):
    chats_folder = os.path.join(CURRENT_DIR, CHAT_FOLDER)
    chat_dirs = [f for f in os.listdir(chats_folder) if os.path.isdir(os.path.join(chats_folder, f))]
    
    registered_chats = []
    for chat_id in chat_dirs:
        # We read the data from the file and try to update the metadata with the current available
        chat_data = load_interaction(chat_folder=os.path.join(CURRENT_DIR, CHAT_FOLDER, str(chat_id)))
        
        # We now try to retrieve the most recent data
        try:
            chat = context.bot.get_chat(int(chat_id))
            chat_data['metadata'] = chat  # We replace the metadata with a (updated) chat instance
            chat_data['error'] = False  # We manually add this field to know that it didn't error
            chat_data['error_msg'] = ''
            
        except Exception as e:
            logger.warning("Can't get updated chat info for {chat_id}: {err_msg}".format(chat_id=chat_id, err_msg=str(e)))
            
            # We fallback to the saved info            
            chat = Chat(**chat_data['metadata'])
            chat_data['metadata'] = chat  # We replace the metadata with a (updated) chat instance
            chat_data['error'] = True  # We manually add this field to know that it didn't error
            chat_data['error_msg'] = str(e)
        
        registered_chats.append(chat_data)
    
    return registered_chats


def bot_status_handler(update, context):  # Prints the status and current users of the bot
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    
    admin_allowed = check_admin_allowed(whitelist=CHAT_WHITELIST, update=update)
    log_msg = 'ALLOWED' if admin_allowed else 'DENIED'
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) sends /STATUS ({message_id}). Query {log_msg}: "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id, user_msg=update.message.text, log_msg=log_msg))
    
    if not admin_allowed:
        return  # We return silently
    
    registered_chats = get_registered_chats(context=context)
    registered_users = [c for c in registered_chats if c['metadata'].type.upper() == 'PRIVATE']
    registered_groups = [c for c in registered_chats if c['metadata'].type.upper() in set(['GROUP', 'SUPERGROUP', 'CHANNEL'])]
    
    status_msg = 'Hi {user_name} (@{t_username}, {user_id}).\nSo far {num_user} users and {num_groups} groups have used the bot:\n\nUSERS:'.format(user_id=user.id, user_name=human_name, t_username=telegram_username, num_user=len(registered_users), num_groups=len(registered_groups))
    
    for registered_user_data in registered_users:
        registered_user = registered_user_data['metadata']
        reg_id = registered_user.id
        reg_human_name = get_human_name(from_user=registered_user)
        reg_t_username = ('@' + registered_user.username) if hasattr(registered_user, 'username') and registered_user.username is not None else None
        
        user_status = '{user_id}: {user_name} ({t_username})'.format(user_id=reg_id, user_name=reg_human_name, t_username=reg_t_username)
        if registered_user_data['error']:
            user_status += ' [ERR]'
        
        user_tokens = registered_user_data['total_tokens']
        user_status += ' Used {:d} tokens ({:.2f}$)'.format(user_tokens, convert_tokens_to_dollars(user_tokens))
        
        status_msg = status_msg + '\n    ' + user_status  # TODO: Add info like last message date or something
    
    status_msg = status_msg + '\n\nGROUPS:'
    
    for registered_group_data in registered_groups:
        registered_group = registered_group_data['metadata']
        reg_id = registered_group.id
        reg_title = registered_group.title
        reg_t_username = ('@' + registered_group.username) if hasattr(registered_group, 'username') and registered_group.username is not None else None
        
        group_status = '{group_id}: {group_title} ({t_username})'.format(group_id=reg_id, group_title=reg_title, t_username=reg_t_username)
        if registered_group_data['error']:
            group_status += ' [ERR]'
        
        group_tokens = registered_group_data['total_tokens']
        group_status += ' Used {:d} tokens ({:.2f}$)'.format(group_tokens, convert_tokens_to_dollars(group_tokens))
        
        for group_member_id, group_member in registered_group_data['members'].items():
            group_user_human_name = get_human_name(from_user=group_member['metadata'])
            group_user_t_username = ('@' + group_member['metadata']['username']) if 'username' in group_member['metadata'] and group_member['metadata']['username'] is not None else None
            group_user_status = '\n        {user_id}: {user_name} ({t_username})'.format(user_id=group_member_id, user_name=group_user_human_name, t_username=group_user_t_username)
            
            group_user_tokens = group_member['total_tokens']
            group_user_status += ' Used {:d} tokens ({:.2f}$)'.format(group_user_tokens, convert_tokens_to_dollars(group_user_tokens))
            group_status += group_user_status
            
        status_msg = status_msg + '\n    ' + group_status  # TODO: Add info like last message date or something
    
    context.bot.send_message(chat_id=chat_id, text=status_msg)


def bot_help_handler(update, context):
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) sends /HELP ({message_id})'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, t_username=telegram_username, message_id=update.message.message_id))
    
    context.bot.send_message(chat_id=chat_id, text="Hello there! I'm {bot_name}, a conversational artificial intelligence. You can query me by starting your messages with \"/AI\".\nFor example: \"/AI What's your name?\"".format(bot_name=DEFAULT_BOT))


def bot_ERROR_handler(update, context):
    if hasattr(update, 'message') and hasattr(update.message, 'text'):
        message_text = update.message.text
    else:
        message_text = None
        
    if hasattr(update, 'message') and hasattr(update.message, 'chat_id'):
        chat_id = update.message.chat_id
    else:
        chat_id = None
    
    if hasattr(update, 'message') and hasattr(update.message, 'message_id'):
        message_id = update.message.message_id
    else:
        message_id = None
    
    if hasattr(update, 'effective_chat') and hasattr(update.effective_chat, 'title'):
        chat_name = update.effective_chat.title
    else:
        chat_name = None
    
    if hasattr(update, 'message') and hasattr(update.message, 'from_user'):
        user_id = update.message.from_user.id
        user_name = get_human_name(from_user=update.message.from_user)
        telegram_username = update.message.from_user.username if hasattr(update.message.from_user, 'username') else None
    else:
        user_id = None
        user_name = None
        telegram_username = None
    
    
    logger.warning('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}, {t_username}) with message ({message_id}) "{user_msg}" caused error "{error_txt}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user_id, user_name=user_name, t_username=telegram_username, message_id=message_id, user_msg=message_text, error_txt=context.error))
    raise context.error


def _read_whitelist_file(file_name):
    with open(file_name, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'CHATS' not in data:
        data['CHATS'] = []
    
    if 'USERS' not in data:
        data['USERS'] = []
    
    if 'ADMINS' not in data:
        data['ADMINS'] = []
    
    # To ease lookups
    data['CHATS'] = set(data['CHATS'])
    data['USERS'] = set(data['USERS'])
    data['ADMINS'] = set(data['ADMINS'])
    
    return data

def _read_prompt_file(file_name):
    with open(file_name, 'r') as f:
        prompt_text = f.read().strip()
    
    return prompt_text


def _read_key(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        key = lines[0].strip()
    
    return key

def main(prompt_file=None, store_conv=False, whitelist_file=None, strict_white_list=False):
    openai.api_key = _read_key(os.path.join('keys', 'openai'))
    telegram_key = _read_key(os.path.join('keys', 'telegram'))
    
    if prompt_file is not None:
        prompt_text = _read_prompt_file(prompt_file)
    else:
        prompt_text = DEFAULT_PROMPT
    
    global CURRENT_PROMPT
    CURRENT_PROMPT = prompt_text
    
    global STORE_CONV
    STORE_CONV = store_conv
    
    global CHAT_WHITELIST
    if whitelist_file is not None:
        CHAT_WHITELIST = _read_whitelist_file(file_name=whitelist_file)
    else:
        CHAT_WHITELIST = None
    
    global STRICT_WHITE_LIST
    STRICT_WHITE_LIST = strict_white_list
    
    
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(telegram_key)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher
	
    dp.add_handler(CommandHandler("AI", bot_ai_handler))
    
    if STABLE_DIFFUSION_PATH is not None and len(STABLE_DIFFUSION_PATH) > 0:
        dp.add_handler(CommandHandler("PIC", bot_pic_handler))
    
    dp.add_handler(CommandHandler("RESET", bot_reset_handler))
    
    dp.add_handler(CommandHandler("HELP", bot_help_handler))
    dp.add_handler(CommandHandler("START", bot_help_handler))  # Just in case someone writes "/start" (apparently people do that)
    
    dp.add_handler(CommandHandler("STATUS", bot_status_handler))  # Only ADMIN can use this command

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, bot_TEXT_handler))

    # log all errors
    dp.add_error_handler(bot_ERROR_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracks a target inside a video and displays the results')
    parser.add_argument('--prompt', metavar='file', default=None, help='file storing the prompt')
    parser.add_argument('--white-list', metavar='file', default=None, help='user/chat whitelist to filter interactions')
    parser.add_argument('--strict-white-list', action='store_true', help='only allows whitelisted users to run the language model')
    parser.add_argument('--store-chats', action='store_true', help='store chats for logging and answering')
    args = parser.parse_args()
    
    # We create the necessary dirs
    os.makedirs(os.path.join(CURRENT_DIR, CHAT_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, LOGS_FOLDER), exist_ok=True)
    
    logs_file = os.path.join(CURRENT_DIR, LOGS_FOLDER, 'log.log')
    
    fh = logging.FileHandler(logs_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    main(prompt_file=args.prompt, store_conv=args.store_chats, whitelist_file=args.white_list, strict_white_list=args.strict_white_list)

