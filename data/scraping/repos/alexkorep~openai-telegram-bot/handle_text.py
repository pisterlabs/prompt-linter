from models.history import get_history, save_history
from models.prompt import get_prompt
from consts import OPENAI_REQUEST_TIMEOUT

# Number of messages to pass to OpenAI (if it fits in the token limit)
HISTORY_LEN = 128
# Model to use
MODEL_NAME = "gpt-3.5-turbo"
# Token limit for the model
MODEL_TOKEN_LIMIT = 4096
# How many tokens we reserve for the history. That means that
# the model response will be cut to MODEL_TOKEN_LIMIT - MODEL_HISTORY_LIMIT tokens.
MODEL_HISTORY_LIMIT = MODEL_TOKEN_LIMIT/2

def num_tokens_from_messages(messages):
    """ Count the number of tokens in the messages """
    words = 0
    for message in messages:
        words += len(message["content"].split())
    # One token is 3/4 of a word
    tokens = int(words / 0.75)
    return tokens

def make_history(chat_dest, text):
    messages = []
    history = get_history(chat_dest, HISTORY_LEN)

    # Messages are in reverse order, so add the current message first
    messages.append({"role": "user", "content": text})
    # The history in reverse order so that the most recent message is first
    for message in history:
        if message["is_user"]:
            role = "user"
        else:
            role = "assistant"
        messages.append({"role": role, "content": message["message"]})
        tokens = num_tokens_from_messages(messages)
        if tokens > MODEL_TOKEN_LIMIT - MODEL_HISTORY_LIMIT:
            # Remove the last message if it puts us over the limit
            messages.pop()
            break
    prompt = get_prompt(chat_dest)
    messages.append({"role": "system", "content": prompt})
    return messages[::-1]

def handle_message_text(bot, openai, body):
    text = body["text"]
    chat_dest = body["chat_dest"]
    messages = make_history(chat_dest, text)

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        request_timeout=OPENAI_REQUEST_TIMEOUT,
    )
    content = response.choices[0].message.content
    bot.send_message(chat_dest, content)

    # TODO rename, maybe make a single call
    save_history(chat_dest, text, True)
    save_history(chat_dest, content, False)

    return "OK"
