from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
import re

channel_layer = get_channel_layer()

# chatgpt imports
import difflib
import openai
from decouple import config

from asgiref.sync import sync_to_async

async def get_gpt_response(message_history):

    ## auth open ai
    async_config = sync_to_async(config)
    openai.api_key = await async_config('open_ai_key')

    async_chat_creation = sync_to_async(openai.ChatCompletion.create)

    messages = [{"role": "system", "content": 'Have a fun conversation in the language of the message. Be a bit crass and amusing. Keep responses short at around three sentences.'}]
    messages += message_history

    response_json = await async_chat_creation(
        model = 'gpt-3.5-turbo-0613',
        messages = messages,
        n = 1,
        max_tokens = 200,
        temperature = 1,
        presence_penalty = 1
    )

    message_response = response_json['choices'][0]['message']['content']
    # if response is 100 tokenns add an mark to indicate the message was cut off.
    if response_json['usage']['completion_tokens'] >= 200:
        message_response += ' –'

    #message_response = 'Response!'
    return message_response

def remedy_corrections(original_message, corrected_message):
    # split text - retain spaces, ignore '-'
    split_orig = re.findall(r'\s?\w+(?:-\w+)*|\s?[^\w\s]', original_message)
    split_corrected = re.findall(r'\s?\w+(?:-\w+)*|\s?[^\w\s]', corrected_message)

    replace_delete_span = '<span class="correction-delete">'
    insert_span = '<span class="correction-insert">'

    # get seq of separated words
    seq = difflib.SequenceMatcher(None, split_orig, split_corrected)
    corrections = ''
    for ops in seq.get_opcodes():
        if ops[0] == 'equal':
            corrections += ''.join(split_corrected[ops[3]:ops[4]])
        if ops[0] == 'replace':
            # otherwise append each word to corrections
            corrections += replace_delete_span + ''.join(split_orig[ops[1]:ops[2]]) + '</span>' + ' ' # space to improve readbility
            corrections += insert_span + ''.join(split_corrected[ops[3]:ops[4]]) + '</span>'
        if ops[0] == 'insert':
            corrections += insert_span + ''.join(split_corrected[ops[3]:ops[4]]) + '</span>'
        if ops[0] == 'delete':
            corrections += replace_delete_span + ''.join(split_orig[ops[1]:ops[2]]) + '</span>'

    return corrections

async def get_gpt_correction(message):

    ## auth open ai
    async_config = sync_to_async(config)
    openai.api_key = await async_config('open_ai_key')

    async_chat_creation = sync_to_async(openai.ChatCompletion.create)

    system_prompt = """
        In the language that the phrase is written, replace it for spelling and grammar. 
        Be sure to add or replace accent marks if necessary. If the use of a word is 
        incorrect replace it. Do not respond to questions.
    """

    response_json = await async_chat_creation(
        model = 'gpt-3.5-turbo-0613',
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': message}
        ],
        n = 1,
        max_tokens = 100,
        temperature = .1,
    )

    message_response = response_json['choices'][0]['message']['content']

    #corrected_message = 'Corrected message'
    corrections = remedy_corrections(message, message_response)

    return corrections
