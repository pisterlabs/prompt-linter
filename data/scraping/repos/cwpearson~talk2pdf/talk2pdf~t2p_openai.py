import hashlib
import json
import sys

import openai

import talk2pdf.config as config
import talk2pdf.utils as utils


def transcribe(path):

    # inputs to OpenAI's translate function
    api_key = config.get(config.KEY_OPENAI_SECRET)
    model = "whisper-1"
    response_format = "verbose_json"

    # hash inputs as key for cache
    h = hashlib.md5()
    h.update(api_key.encode('utf-8'))
    h.update(model.encode('utf-8'))
    utils.eprint(f"==== open {path} for hashing...")
    with open(path, 'rb') as f:
        h.update(f.read())
    h.update(response_format.encode('utf-8'))
    digest = h.hexdigest()
    utils.eprint(f"==== transcribe hash is {digest}")
    cached_response_path = config.get(config.KEY_CACHE_DIR) / f"{digest}.json"

    # reach cached reponse, or cache a new response
    if cached_response_path.is_file():
        utils.eprint(
            f"==== retrieving cached response from {cached_response_path}")
        with open(cached_response_path, 'r') as f:
            transcript = json.loads(f.read())
    else:
        utils.eprint(f"==== open {path} for transcription...")
        with open(path, 'rb') as f:
            openai.api_key = api_key
            transcript = openai.Audio.translate(
                model, f, response_format=response_format)
        utils.eprint(f"==== caching response @ {cached_response_path}")
        config.get(config.KEY_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with open(cached_response_path, 'w') as f:
            f.write(json.dumps(transcript))

    # return the result
    return transcript


def clean(text):

    api_key = config.get(config.KEY_OPENAI_SECRET)
    model = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You split text into paragraphs."},
        {"role": "user", "content": f"Split the following text into paragraphs; DO NOT REMOVE TEXT, DO NOT LABEL PARAGRAPHS:\n{text}"}
    ]

    h = hashlib.md5()
    h.update(api_key.encode('utf-8'))
    h.update(model.encode('utf-8'))
    for msg in messages:
        h.update(msg["role"].encode('utf-8'))
        h.update(msg["content"].encode('utf-8'))
    digest = h.hexdigest()
    utils.eprint(f"==== clean hash is {digest}")
    cached_response_path = config.get(config.KEY_CACHE_DIR) / f"{digest}.json"

    if cached_response_path.is_file():
        utils.eprint(
            f"==== retrieving cached response from {cached_response_path}")
        with open(cached_response_path, 'r') as f:
            response = json.loads(f.read())
    else:
        utils.eprint(
            f"==== {cached_response_path} did not exist. Submitting to OpenAI...")
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        utils.eprint(f"==== caching response @ {cached_response_path}")
        config.get(config.KEY_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with open(cached_response_path, 'w') as f:
            f.write(json.dumps(response))

    content = response['choices'][0]['message']['content'].strip()
    total_tokens = int(response['usage']['total_tokens'])

    if len(content) < len(text) * 0.95:
        utils.eprint("==== dropped too much text during cleaning!")
        utils.eprint(messages)
        sys.exit(1)
        return text
    else:
        return content
