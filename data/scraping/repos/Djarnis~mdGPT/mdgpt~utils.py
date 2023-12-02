import os
import pycountry
import openai
import glob
import json

from pathlib import Path

from mdgpt.models import PromptConfig
from mdgpt.models import ChatGPTModel


def urlize(text):
    if text.endswith('index.md'):
        text = text[:-9]
    return text


def log_usage(action, target, file, prompt_tokens, completion_tokens):
    logger = Path(f'./.mdgpt-usage/{action}.csv')
    if not logger.exists():
        logger.parent.mkdir(parents=True, exist_ok=True)
        logger.write_text('target;file;prompt_tokens;completion_tokens\n')

    with logger.open('a') as f:
        f.write(f'{target};{file};{prompt_tokens};{completion_tokens}\n')


def get_chat_response(messages, model='gpt-3.5-turbo', temperature=0.2, max_tokens=1024 * 2, n=1):
    if os.getenv('OPENAI_API_KEY') is None:
        print('Please set OPENAI_API_KEY environment variable.')
        exit(1)

    openai.api_key = os.getenv('OPENAI_API_KEY')
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            max_tokens=max_tokens,
            n=n,
            stop=None,
            temperature=temperature,
            messages=messages,
        )

        message = completions.choices[0].message.content
        usage = completions.usage.to_dict()

        return message, usage

    except openai.error.AuthenticationError as e:
        print(f'OpenAI AuthenticationError: {e}')
        raise e

    except openai.error.OpenAIError as e:
        print(f'OpenAI Error: {e}')
        raise e

    except Exception as e:
        print(f'An error occurred: {e}')
        raise e


def get_gpt_options(gpt_model: ChatGPTModel):
    options = {}

    if gpt_model is None:
        return options

    if gpt_model.engine:
        options['model'] = gpt_model.engine
    if gpt_model.temperature:
        options['temperature'] = gpt_model.temperature
    if gpt_model.max_tokens:
        options['max_tokens'] = gpt_model.max_tokens

    return options


def get_language_name(lang_code):
    lang = pycountry.languages.get(alpha_2=lang_code)
    if lang is None:
        print(f'Language {lang_code} not found.', lang)
        exit(1)

    return lang.name


def get_url_map(prompt_cfg: PromptConfig):
    files = get_markdown_files(Path(prompt_cfg.ROOT_DIR, prompt_cfg.SOURCE_DIR if prompt_cfg.SOURCE_DIR else prompt_cfg.LANGUAGE))

    if prompt_cfg.ONLY_INDEXES:
        url_hash = {urlize(f): '' for f in files if f.endswith('index.md')}
    else:
        url_hash = {urlize(f): '' for f in files}

    return url_hash


def get_url_matrices(cfg: PromptConfig):
    # Build url maps for each target language
    lang_matrix, url_matrix, missing_matrix = {}, {}, {}
    for target in cfg.TARGET_LANGUAGES:
        url_hashes = get_url_map(cfg)
        url_map, missing = get_json_to_translate(cfg, url_hashes, target)

        url_matrix[target] = url_map
        missing_matrix[target] = missing
        lang_matrix[target] = get_lang_dict(target)

    return lang_matrix, url_matrix, missing_matrix


def get_json_to_translate(prompt_cfg: PromptConfig, json_dict, target):
    filename = f'{prompt_cfg.LANGUAGE}_{target}.json'
    src_file = Path(f'{prompt_cfg.ROOT_DIR}/.mdgpt-urls/{filename}')

    if src_file.exists() and prompt_cfg.IGNORE_CACHE is False:
        src_json = json.loads(src_file.read_text())

        for k, v in json_dict.items():
            if src_json.get(k):
                json_dict[k] = src_json.get(k)

    if json_dict.get(''):
        del json_dict['']

    missing = {}
    for k, v in json_dict.items():
        if k == '':
            # print('This is empty', k, v)
            continue

        if v is None or v == '':
            # print('This is none or empty', k, v)
            missing[k] = v

    # print('missing', len(missing), missing)
    return json_dict, missing


def get_lang_dict(lang):
    return {
        'code': lang,
        'name': get_language_name(lang),
        'dir': lang,
    }


def get_markdown_files(path: Path):
    files = []
    for filepath in glob.iglob(f'{path}/**/*.md', recursive=True):
        relative_path = filepath[len(f'{path}') :]
        if relative_path.endswith('README.md'):
            continue
        files.append(relative_path.lstrip('/'))
    files.sort()
    return files


class DefaultDictFormatter(dict):
    def __missing__(self, key):
        return MissingAttrHandler('{{{}'.format(key))


class MissingAttrHandler(str):
    def __init__(self, format):
        self.format = format

    def __getattr__(self, attr):
        return type(self)('{}.{}'.format(self.format, attr))

    def __repr__(self):
        return MissingAttrHandler(self.format + '!r}')

    def __str__(self):
        return MissingAttrHandler(self.format + '!s}')

    def __format__(self, format):
        if self.format.endswith('}'):
            self.format = self.format[:-1]
        return '{}:{}}}'.format(self.format, format)
