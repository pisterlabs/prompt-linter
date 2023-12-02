'''
Code for querying fine-tuned GPT models.
'''

import logging
import time

import openai


async def ask_ft_gpt(query, model=None):
    if not model:
        raise ValueError('Must specify fine tuned model to use for query')

    logging.info(f'Using {model} to generate completion for query: {query}')

    fetch_st = time.time()
    try:
        result = await openai.Completion.acreate(
            model=model,
            prompt=query,
            max_tokens=256,
            temperature=0.2,
        )
    except Exception as err:
        logging.error(f'Error querying fine-tuned GPT model: {err}')
        raise err
    fetch_et = time.time()
    logging.info(f'Custom model response fetched in {(fetch_et - fetch_st):.2f} seconds')

    result = result.choices[0].text.split('\n')
    result = result[0] if result[0] else result[1]

    return {'answer': result.strip()}
