import datetime
import functools
import json
import os
import logging
import re
import time
import traceback
from datetime import date
from textwrap import dedent

import tiktoken
from pyquery import PyQuery
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai
from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError

# setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()
app = App()
my_user_id = app.client.auth_test().data["user_id"]
openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_TEXT_PARAMS = {
    'model': os.getenv("MODEL", "gpt-4"),
    'temperature': 0.7,
    # 'max_tokens': 250,
    # 'top_p': 1,
    # 'frequency_penalty': 0,
    # 'presence_penalty': 0,
}
MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', '4096'))
OPENAI_IMG_PARAMS = {
    "n": 1,
    "size": "1024x1024",
}
BOT_NAME = os.getenv('BOT_NAME', 'AbbyLarby')
HIDDEN_PROMPT_URL = os.getenv('HIDDEN_PROMPT_URL')
tokenizer = tiktoken.encoding_for_model(OPENAI_TEXT_PARAMS['model'])

### helpers ###

def get_token_count(s):
    return len(tokenizer.encode(s))

def ttl_cache(seconds=60):
    """ via https://stackoverflow.com/a/50866968/307769 """
    ttl = datetime.timedelta(seconds=seconds)
    def wrap(func):
        cache = {}
        @functools.wraps(func)
        def wrapped(*args, **kw):
            now = datetime.datetime.now()
            key = tuple(args), frozenset(kw.items())
            if key not in cache or now - cache[key][0] > ttl:
                value = func(*args, **kw)
                cache[key] = (now, value)
            return cache[key][1]
        return wrapped
    return wrap

def get_text(messages, **extra_params):
    if type(messages) is str:
        messages = [{"role": "user", "content": messages}]
    response = openai.ChatCompletion.create(
        messages=messages,
        **{**OPENAI_TEXT_PARAMS, **extra_params}
    )
    logger.debug(f"OpenAI response: {response}")
    return response['choices'][0]['message']['content']

def get_image(prompt, **extra_params):
    response = openai.Image.create(prompt=prompt, **{**OPENAI_IMG_PARAMS, **extra_params})
    logger.debug(f"OpenAI response: {response}")
    return response['data'][0]['url']

def block_text(text):
    """Return simple text string formatted for Slack block."""
    return {
        "type": "plain_text",
        "text": text,
    }

@ttl_cache(60*60)
def get_team_fields():
    team_profile = app.client.team_profile_get()
    return {f["id"]: f["label"] for f in team_profile["profile"]["fields"]}

@ttl_cache(60)
def id_to_user_info(user_id):
    team_fields = get_team_fields()
    try:
        user_info = app.client.users_profile_get(user=user_id)["profile"]
    except SlackApiError:
        # for unknown user (typically cross-slack convo), just say "unknown" for now
        return {"first_name": "Unknown Name", "real_name": "Unknown Name", "display_name": "Unknown Name"}
    for label_id, label in team_fields.items():
        user_info[label] = user_info["fields"].get(label_id, {'value': ''})['value']
    return user_info

@ttl_cache(60*5)
def get_hidden_prompt():
    if HIDDEN_PROMPT_URL:
        doc = PyQuery(url=HIDDEN_PROMPT_URL)
        prompt = "\n\n".join(text for p in doc('p').items() if (text := p.text().strip()) != "" and not text.startswith('#')).strip()
        if not prompt:
            raise ValueError(f"No prompt found in HIDDEN_PROMPT_URL.")
        return prompt
    return f"You are {BOT_NAME}, a friendly, helpful AI bot."

def hydrate_user_ids(text):
    def id_to_name(m):
        user = id_to_user_info(m[1])
        return f'@{user["display_name"] or user["real_name"]}'
    return re.sub(r'<@(U[A-Z0-9]{10})>', id_to_name, text)

def is_command(text, command, is_dm):
    return text == f"@AbbyLarby {command}" or (is_dm and text == command)

def readable_timedelta(seconds):
    # via https://codereview.stackexchange.com/a/245215
    data = {}
    data['days'], remaining = divmod(seconds, 86_400)
    data['hours'], remaining = divmod(remaining, 3_600)
    data['minutes'], data['seconds'] = divmod(remaining, 60)

    time_parts = [f'{round(value)} {name}' for name, value in data.items() if value > 0]
    if time_parts:
        return ' '.join(time_parts)
    else:
        return 'less than 1 second'

def respond_errors(func):
    """Catch errors in handler and post to slack."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = f"```Exception handling request:\n{traceback.format_exc()}\n```"
            if 'respond' in kwargs:
                kwargs['respond'](err_str, response_type="ephemeral")
            elif 'say' in kwargs:
                kwargs['say'](err_str)
            raise
    return wrapper

def get_system_prompt(bios):
    bios = '\n* '.join(bios.values())
    return f"""
{get_hidden_prompt()}

Today is {date.today().strftime("%A, %B %-d, %Y.")}

You can refer to users by name. This is what you know about the users:
* {bios}
You should not invent any other users! You're only talking to these users.

User names will be separated from their comments by [name_separator].
""".strip()

### views ###

@app.command("/ai")
@respond_errors
def ai(ack, respond, command):
    logger.debug(command)
    ack()

    response_type = "ephemeral"
    img_prompt = False
    prompt = command['text']

    # help command
    if prompt == "help":
        respond(dedent(f"""
            Available commands:
            * `/ai <prompt>`: Show the {OPENAI_TEXT_PARAMS['model']} model's response to `<prompt>`. Visible only to you.
            * `/ai img <prompt>`: Show the Dall-E image generated by `<prompt>`. Visible only to you.
            * `/ai say <prompt>` or * `/ai say img <prompt>`: immediately post the result to the channel.
        """))
        return

    # parse 'say' and 'img' from prompt
    if prompt.split(maxsplit=1)[0] == "say":
        response_type = "in_channel"
        prompt = prompt.split(maxsplit=1)[1]
    if prompt.split(maxsplit=1)[0] == "img":
        img_prompt = True
        prompt = prompt.split(maxsplit=1)[1]

    # generate image or text response
    formatted_prompt = f"{command['user_name']} asked: /ai {command['text']}"
    if img_prompt:
        response = get_image(prompt)
        blocks = [{
            "type": "image",
            "title": block_text(formatted_prompt),
            "image_url": response,
            "alt_text": f"Dall-E image generated for the prompt {prompt}",
        }]
    else:
        response = get_text(prompt)
        blocks = [
            {"type": "context", "elements": [block_text(formatted_prompt)]},
		    {"type": "section", "text": block_text(response)},
        ]

    # show "Post publicly" button if message not already public, and we're not in a DM where we can't post
    if response_type == "ephemeral" and command["channel_name"] != "directmessage":
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": block_text("Post publicly"),
                    "value": json.dumps({"text": formatted_prompt, "blocks": blocks}),
                    "action_id": "public_repost"
                }
            ]
        })

    respond(formatted_prompt, response_type=response_type, blocks=blocks)

@app.action("public_repost")
@respond_errors
def public_repost(respond, ack, payload, say):
    """Handle 'Post publicly' button."""
    ack()
    to_repost = json.loads(payload['value'])
    say(to_repost["text"], response_type="in_channel", blocks=to_repost["blocks"])
    respond(text='', replace_original=True, delete_original=True)

@app.event("app_mention")
@respond_errors
def handle_mention(say, ack, payload):
    """Handle @ mention of AbbyLarby."""
    ack()
    handle_conversation(say, payload, is_dm=False)

@app.event("message")
@respond_errors
def handle_dm(ack, payload, say):
    """Handle conversations with the app itself."""
    ack()
    handle_conversation(say, payload, is_dm=True)

def handle_conversation(say, payload, is_dm=False):
    users_in_convo = {my_user_id: id_to_user_info(my_user_id)}
    latest_message = hydrate_user_ids(payload["text"])

    # if we were mentioned in a thread, say() should respond in the thread
    if 'thread_ts' in payload:
        wrapped_say = say
        say = lambda *args, **kwargs: wrapped_say(*args, **kwargs, thread_ts=payload["thread_ts"])

    # handle 'help' command
    if is_command(latest_message, "help", is_dm):
        say(dedent(f"""
            I'm {BOT_NAME}, a friendly, helpful AI bot. You can just talk to me, or I'll do special things if you send one of these messages:
            * `reset`: I'll ignore anything we said before this message.
            * `prompt`: I'll show the entire prompt I would have used to generate a response.
        """))
        return

    # handle 'reset' command
    if is_command(latest_message, "reset", is_dm):
        if is_dm:
            say(f"Hi! I'm {BOT_NAME}.")
        return

    # fetch previous slack messages in conversation, and turn into prompts
    if 'thread_ts' in payload:
        messages = app.client.conversations_replies(channel=payload["channel"], ts=payload["thread_ts"])
        messages = reversed(messages.data["messages"])
    else:
        messages = app.client.conversations_history(channel=payload["channel"])
        messages = messages.data["messages"]
    prompt_messages = []
    tokens_remaining = int(MODEL_MAX_TOKENS * .75)
    bios = {}
    system_prompt = get_system_prompt(bios)
    for message in messages:
        if not 'user' in message:
            continue
        users_in_convo[message['user']] = user_info = id_to_user_info(message['user'])
        if message['user'] == my_user_id:
            role = 'assistant'
        else:
            role = 'user'
            if message['user'] not in bios:
                # recalculate system_prompt each time we add a new user, so we can accurately check token count
                bios[message['user']] = f'First name: {user_info.get("first_name", user_info.get("real_name"))}. This person also goes by: {user_info.get("display_name")}. Pronouns: {user_info.get("pronouns", "they/them")}. What you know about this user: "{user_info.get("Info for AbbyLarby", "they like ducks!")}"'
                system_prompt = get_system_prompt(bios)
        content = hydrate_user_ids(message['text'])
        if not content.strip():
            continue

        # handle reset keyword
        if is_command(content, "reset", is_dm):
            break

        # skip previous prompt inspections
        if is_command(content, "prompt", is_dm):
            continue

        content = f"{user_info.get('first_name', user_info['real_name'])} [name_separator] {content}"

        # enforce length limit -- excessive prompt length will cause API error
        tokens_remaining -= get_token_count(content)
        if tokens_remaining - get_token_count(system_prompt) <= 0:
            break

        prompt_messages.append({"role": role, "content": content})

    # we go through messages backwards to the start, so end up with messages in reverse order
    prompt_messages.reverse()

    # add system prompt
    prompt_messages.insert(0, {"role": "system", "content": system_prompt})

    if is_command(latest_message, "prompt", is_dm):
        response = json.dumps(OPENAI_TEXT_PARAMS, indent=4)+"\n\n"+json.dumps(prompt_messages, indent=4)
        app.client.files_upload(
            channels=payload["channel"],
            filename="prompt.json",
            content=response,
        )
    else:
        response = get_text(prompt_messages)
        response = response.rsplit('[name_separator]', 1)[-1]
        say(response, response_type="in_channel")


if __name__ == "__main__":
    handler = SocketModeHandler(app)
    handler.start()