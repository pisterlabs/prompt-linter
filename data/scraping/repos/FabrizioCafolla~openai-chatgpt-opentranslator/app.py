import pathlib
import sys
from decimal import Decimal

import click
import openai

_VERBOSITY = 1

# Calculate token https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

# Link https://beta.openai.com/docs/models/gpt-3
_DEFAULT_ENGINE = 'text-curie-001'
_AVAILABLE_ENGINES = ['text-ada-001', 'text-babbage-001',
                      'text-curie-001', 'text-davinci-003']

# Link https://openai.com/api/pricing
# 750 words = 1000 tokens
_WORDS = 750
_TOKEN = 1000
_PRICING_PER_1K_TOKENS = {
    'text-ada-001': 0.0004,
    'text-babbage-001': 0.0005,
    'text-curie-001': 0.0020,
    'text-davinci-003': 0.0200
}

# Link https://beta.openai.com/docs/models/gpt-3
_TOKEN_LIMITS_PER_REQUEST = {
    'text-ada-001': 2048,
    'text-babbage-001': 2048,
    'text-curie-001': 2048,
    'text-davinci-003': 4000
}


def set_verbosity(verbose: int):
    if verbose > 0 and verbose <= 3:
        globals()['_VERBOSITY'] = verbose


def render_file(filepath: str) -> str:
    with open(str(pathlib.Path(filepath)), 'r') as f:
        return f.read()


def calculate_pricing(engine: str, tokens: int) -> Decimal:
    pricing = Decimal((tokens / _TOKEN) * _PRICING_PER_1K_TOKENS[engine])
    return round(pricing, 4)


def verbose(level: str):
    import functools

    def actual_decorator(func):
        def neutered(*args, **kw):
            return

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return (
                func(*args, **kwargs)
                if level <= _VERBOSITY
                else neutered
            )

        return wrapper

    return actual_decorator


@verbose(level=3)
def debug(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


@verbose(level=2)
def info(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


@verbose(level=1)
def echo(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


@click.command()
@click.option('--translate', type=str, required=True)
@click.option('--text', default=None, help='Raw text')
@click.option('--filepath', default=None, help='File path containing the text')
@click.option('--engine', default=_DEFAULT_ENGINE, type=click.Choice(_AVAILABLE_ENGINES), help='GPT-3 engine')
@click.option('--temperature', default=0.7, help='Higher values means the model will take more risks. Values 0 to 1')
@click.option('--max-token', default=256, help='The maximum number of tokens to generate in the completion.')
@click.option('-v', '--verbose', default=1, count=True)
def main(translate, text, filepath, engine, max_token, temperature, verbose):
    try:
        set_verbosity(verbose)

        prefix = f'Translate the following text into {translate}'

        if filepath:
            text = render_file(filepath=filepath)

        if not text:
            raise Exception('No text provided to the command')

        text = f'{prefix}:\n{text}'

        request_token = round(_WORDS / len(text))
        request_pricing = calculate_pricing(engine, request_token)
        debug(f'# Request cost: ${request_pricing}')
        debug(
            f'# Request token limit: {request_token}/{_TOKEN_LIMITS_PER_REQUEST[engine]}')

        if request_token > _TOKEN_LIMITS_PER_REQUEST[engine]:
            raise Exception(
                f'The required tokens {request_token} are greater than the limit of {_TOKEN_LIMITS_PER_REQUEST[engine]}')

        response = openai.Completion.create(
            model=engine,
            prompt=text,
            temperature=temperature,
            max_tokens=max_token,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_token = response.get('usage', {}).get('completion_tokens', 0)
        response_pricing = calculate_pricing(engine, response_token)
        debug(f'# Response cost: ${response_pricing}')
        debug(f'# Response tokens: {response_token}')
        debug(f'# Total tokens: {request_token + response_token}')
        debug(f'# Total cost: ${request_pricing + response_pricing}')

        translated_text = ''

        if choices := response.get('choices', []):
            if len(choices) > 0:
                translated_text = choices[0]['text']
    except Exception as e:
        echo(f'[ERR] {str(e)}')
        sys.exit(1)
    else:
        info('\n# Input:')
        info(text)
        info('\n# Output:')
        echo(translated_text.strip())


if __name__ == '__main__':
    sys.exit(main())
