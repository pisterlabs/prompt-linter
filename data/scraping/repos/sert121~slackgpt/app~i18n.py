from typing import Optional

import openai
from slack_bolt import BoltContext

from .openai_ops import GPT_3_5_TURBO_0301_MODEL

# All the supported languages for Slack app as of March 2023
_locale_to_lang = {
    "en-US": "English",
    "en-GB": "English",
    "de-DE": "German",
    "es-ES": "Spanish",
    "es-LA": "Spanish",
    "fr-FR": "French",
    "it-IT": "Italian",
    "pt-BR": "Portuguese",
    "ru-RU": "Russian",
    "ja-JP": "Japanese",
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "ko-KR": "Korean",
}


def from_locale_to_lang(locale: Optional[str]) -> Optional[str]:
    if locale is None:
        return None
    return _locale_to_lang.get(locale)


_translation_result_cache = {}


def translate(*, openai_api_key: str, context: BoltContext, text: str) -> str:
    # lang = from_locale_to_lang(context.get("locale"))
    # if lang is None or lang == "English":
    #     return text

    # cached_result = _translation_result_cache.get(f"{lang}:{text}")
    # if cached_result is not None:
    #     return cached_result
    response = openai.ChatCompletion.create(
        api_key=openai_api_key,
        model=GPT_3_5_TURBO_0301_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an SQL analyst who is really good at translation of Natural language to SQL. Given a question and a table schema construct a valid SQL query. ",
            },
            {
                "role": "user",
                "content": f"You are an SQL analyst who is really good at translation of Natural language to SQL. Given a question and a table schema construct a valid SQL query.{text}",
            },
        ],
        top_p=1,
        n=1,
        max_tokens=1024,
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias={},
        user="system",
        api_base=context.get("OPENAI_API_BASE"),
        api_type=context.get("OPENAI_API_TYPE"),
        api_version=context.get("OPENAI_API_VERSION"),
        deployment_id=context.get("OPENAI_DEPLOYMENT_ID"),
    )
    translated_text = response["choices"][0]["message"].get("content")
    # _translation_result_cache[f"{lang}:{text}"] = translated_text
    return translated_text


