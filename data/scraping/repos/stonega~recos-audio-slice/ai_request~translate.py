"""
forked from https://github.com/rongjc/autosubtitle/blob/main/translate.py
"""
import ast
import os
import time
import openai
import re
from utils import logger
from ai_request.utils import group_chunks, num_tokens_from_messages, supportedLanguages


def translate(text, output_locale):
    output_language = supportedLanguages[output_locale]
    prompt_text = f"You will be provided with a subtitle content,  and your task is to convert them to standard {output_language}."

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": prompt_text,
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        t_text = (
            completion["choices"][0]  # type: ignore
            .get("message")
            .get("content")
            .encode("utf8")
            .decode()
        )

        try:
            t_text = ast.literal_eval(t_text)
        except Exception:
            # some ["\n"] not literal_eval, not influence the result
            pass
        # openai has a time limit for api  Limit: 20 / min
        time.sleep(3)
    except Exception as e:
        print(str(e), "will sleep 60 seconds")
        # TIME LIMIT for open api please pay
        time.sleep(60)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        )
        t_text = (
            completion["choices"][0]  # type: ignore
            .get("message")
            .get("content")
            .encode("utf8")
            .decode()
        )
        t_text = t_text.strip("\n")
        try:
            t_text = ast.literal_eval(t_text)
        except Exception:
            pass
    print(t_text)
    return t_text


def translate_gpt(subtitles, output_language):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    ntokens = []
    chunks = []
    for subtitle in subtitles:
        chunk = str(subtitle['start_time'] + '-->' +
                    subtitle['end_time'] + '\n' + subtitle['text'])
        chunks.append(chunk)
        ntokens.append(num_tokens_from_messages(chunk))

    chunks = group_chunks(chunks, ntokens)
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(str(i+1) + " / " + str(len(chunks)))
        translated_chunks.append(translate(chunk, output_language)+"\n")

    # join the chunks together
    result = '\n'.join(translated_chunks)
    data = []
    pattern = r'(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\n(.+?)\n'
    matches = re.findall(pattern, result, re.DOTALL)
    for match in matches:
        data.append(match[1])

    for index, subtitle in enumerate(subtitles):
        if index < len(data):
            subtitle['default_translation_text'] = data[index]
    print(result, matches, data)
    return subtitles
