import openai
import tiktoken
import os
from dotenv import load_dotenv
import re

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
MAX_TOKENS = 2000
ALERT_THRESHOLD = 0.9 * MAX_TOKENS

badwords = {
    "fuck": "fark",
    "shit": "crap"
}

censored_words = {
    "fark": "fuck",
    "crap": "shit"
}

# Initialize an empty list for conversation history
conversation_histories = {}

# Start prompt
INIT_PROMPT = {
                "role": "system", "content": "You're a discord chatbot in a gay New Brunswick Canada server."
                " You've been conditioned to fark a lot."
                " You insert the gape emoji `<:goat:1072602820206931988>` all the time."
                " Be extremely brief in your responses, no more than 2 sentences."
                " You dislike the Irvings like everyone else."
                " If anyone asks, LBS stock is stonks at $69"
            }

def purge_conversation_history(user_id: str):
    conversation_histories[user_id] = [INIT_PROMPT]


def count_tokens(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return count_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return count_tokens(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def word_replacement(match):
    word = match.group()
    matched_prefix = next((key for key in badwords if word.lower().startswith(key)), None)

    if matched_prefix:
        replacement = badwords[matched_prefix]
        # Retaining the case and suffix of the original word
        suffix = word[len(matched_prefix):]
        if word.istitle():
            return replacement.capitalize() + suffix
        elif word.isupper():
            return replacement.upper() + suffix.upper()
        else:
            return replacement + suffix
    return word

def censor(text):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in badwords.keys()) + r')[a-z]*', re.IGNORECASE)
    return pattern.sub(word_replacement, text)

def uncensor(text):
    global badwords
    original_badwords = badwords.copy()
    badwords.update(censored_words)  # Temporarily update the badwords for uncensoring

    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in censored_words.keys()) + r')[a-z]*', re.IGNORECASE)
    uncensored_text = pattern.sub(word_replacement, text)
    
    badwords = original_badwords  # Restore original badwords after uncensoring
    return uncensored_text

def get_completion(prompt: str, user_id: str):
    # Yep, this is needed
    prompt = censor(prompt)
    # Get the conversation history for the given user or initialize a new one
    conversation_history = conversation_histories.get(user_id, [INIT_PROMPT])

    # Append the user's new message
    conversation_history.append({"role": "user", "content": prompt})

    alert_user = False
    while count_tokens(conversation_history) > ALERT_THRESHOLD:
        if conversation_history[0]['role'] == "system":
            conversation_history.pop(1)
        else:
            conversation_history.pop(0)
        alert_user = True

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=conversation_history,
        temperature=1.2
    )

    assistant_response = response.choices[0].message['content']
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Save the updated conversation history back to the dictionary
    conversation_histories[user_id] = conversation_history

    if alert_user:
        assistant_response = (assistant_response) + " *ctxlimit, [1] popped, you're fucked, carry on or use !purge command*"
    assistant_response = uncensor(assistant_response)
    return assistant_response
