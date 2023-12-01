import ast
import os
import time
import openai
from utils import logger, parse_srt
from ai_request.utils import group_chunks, num_tokens_from_messages


def fix(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt_text="You will be provided with a subtitle content, and your task is to combine two subtitle items if the item sentence is not complete.  Then correct any spelling discrepancies in the content. Then if the slash word is inside a url, convert it to /."
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
        logger.info(t_text)

        try:
            t_text = ast.literal_eval(t_text)
        except Exception:
            pass
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
    return t_text


def fix_subtitle(subtitles):
    ntokens = []
    chunks = []
    for subtitle in subtitles:
        logger.info(subtitle)
        chunk = str(subtitle["id"]) + '\n' + subtitle["time"] + '\n' + subtitle["text"]
        chunks.append(chunk)
        ntokens.append(num_tokens_from_messages(chunk))
    chunks = group_chunks(chunks, ntokens)
    fixed_chunks = []
    for chunk in chunks:
        logger.info(chunk)
        fixed_chunks.append(fix(chunk))

    # join the chunks together
    result = '\n'.join(fixed_chunks)
    logger.info(result)
    return parse_srt(result)
