
import os
import sys
current_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, '..'))

sys.path.insert(0, root_directory)

import openai as openai
import pandas as pd
import re
import logging
import datetime
from fuzzywuzzy import process
from typing import List, Tuple

from core.config import Config

openai.api_key = Config.OPENAI_API_KEY

_df = pd.read_csv(Config.COIN_SYMBOLS)


async def get_completion(prompt : list):
    """Call OpenAI's Chat API and return the completion message."""
    response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = prompt,
        max_tokens = 200,
        temperature = 0.5  
    )

    completion = f"{response['choices'][0]['message']['content']}"
    return completion


async def get_symbol_fuzzy(names: list) -> list:
    # Read the CSV file into a DataFrame

    symbols = []
    for name in names:
        # Combine 'Name' and 'Symbol' columns for fuzzy matching
        combined = pd.concat([_df['Name'], _df['Symbol']]).reset_index(drop=True)
        # Find the closest matching name or symbol using fuzzy string matching
        best_match_result = process.extractOne(name, combined)

        # Unpack the tuple containing the best match and the match score
        best_match, match_score = best_match_result[0], best_match_result[1]
        logging.info(f'best_match: {best_match}, match_score: {match_score}')
        # If the match score is above a certain threshold (e.g., 80), get the symbol
        if match_score > 80:
            row = _df.loc[(_df['Name'] == best_match) | (_df['Symbol'] == best_match)]
            symbol = row['Symbol'].values[0]
            symbols.append(symbol)
        else:
            symbols.append(None)

    return symbols


async def extract_time_interval(query: str) -> Tuple[List[datetime.datetime], List[str]]:
    """Returns the datetime interval [start, end] and token list ["token"]."""
    
    # Read the prompt from the file if it hasn't been read already
    if not hasattr(extract_time_interval, '_prompt'):
        with open(Config.PROMPT_FILE, 'r') as f:
            extract_time_interval._prompt = f.read()

    time_reference = datetime.datetime.today()
    time_reference_prompt_str = f'\nRemember the current time is {time_reference}.'

    message = [
        {"role": "user", "content": extract_time_interval._prompt + "\"" + query + "\"" + time_reference_prompt_str}
    ]
    
    completion = await get_completion(message)

    logging.debug(f'completion: {completion}')

    # Extract the time interval list
    time_interval_list = await _extract_time_interval_list(completion)

    # Extract the symbol list
    symbol_list = await _extract_symbol_list(completion)

    return time_interval_list, symbol_list

    
async def _extract_time_interval_list(completion: str) -> List[datetime.datetime]:
    """Extracts the time interval list from the given completion."""

    time_intervals = re.findall(r"\[(.*?)\]", completion)
    time_interval_list = []

    for time_interval_str in time_intervals[:2]:
        datetime_expression = eval(time_interval_str)
        time_interval_list.append(datetime_expression)

    logging.debug(f'time_interval_list: {time_interval_list}')
    return time_interval_list

async def _extract_symbol_list(completion: str) -> List[str]:
    """Extracts the symbol list from the given completion."""

    symbol_match = re.findall(r"\[(.*?)\]", completion)[2]
    symbol_list = [eval(x) for x in symbol_match.split(",")]
    
    logging.debug(f'symbol_list: {symbol_list}')
    return symbol_list


    
