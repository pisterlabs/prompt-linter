import pandas as pd
import openai_api
import dataset
from datetime import datetime
import asyncio
import numpy as np
import re


def check_and_create_columns(df, column_names):
    for column_name in column_names:
        if column_name not in df.columns:
            df[column_name] = None


def fix_empty(df):
    # Preparación
    MAX_PROMPT = 5
    for index in range(len(df)):
        print(f"{index} of {len(df)}")
        for number_prompt in range(1, MAX_PROMPT):
            for model in ['davinci', 'gpt3']:
                if (isinstance(df.loc[index, f'Prompt_{number_prompt}_{model}'], float) and np.isnan(df.loc[index, f'Prompt_{number_prompt}_{model}'])) or (df.loc[index, f'Prompt_{number_prompt}_{model}'] is None):
                    word = df.loc[index, 'Word']
                    if model == 'davinci':
                        df.loc[index, f'Prompt_{number_prompt}_davinci'] = asyncio.run(openai_api.ask_completion(word, type=number_prompt, timeout=120))
                    elif model == 'gpt3':
                        df.loc[index, f'Prompt_{number_prompt}_gpt3'] = asyncio.run(openai_api.send_message_openai(word, type=number_prompt, timeout=120))
    return df 


async def loop_through(df):
    MAX_PROMPT = 5
    temp1 = [f'Prompt_{number_prompt}_gpt3' for number_prompt in range(1, MAX_PROMPT)]
    temp2 = [f'Prompt_{number_prompt}_davinci' for number_prompt in range(1, MAX_PROMPT)]
    check_and_create_columns(df, temp1 + temp2)
    paso = 50
    for inicio, fin in zip(range(0, len(df), paso), range(paso, len(df)+paso, paso)):
        for number_prompt in range(1, MAX_PROMPT):
            df.loc[inicio:fin, f'Prompt_{number_prompt}_gpt3'] = await asyncio.gather(*(openai_api.send_message_openai(word, type=number_prompt) for word in df.loc[inicio:fin, 'Word']))
            df.loc[inicio:fin, f'Prompt_{number_prompt}_davinci'] = await asyncio.gather(*(openai_api.ask_completion(word, type=number_prompt) for word in df.loc[inicio:fin, 'Word']))

    return df


def binary(df):
    MAX_PROMPT = 5
    temp1 = [f'Prompt_{number_prompt}_gpt3' for number_prompt in range(1, MAX_PROMPT)]
    temp2 = [f'Prompt_{number_prompt}_davinci' for number_prompt in range(1, MAX_PROMPT)]
    columns = temp1 + temp2
    for column in columns:
        df[f'{column}_binario'] = df[column].str.contains('|'.join('yes'), case=False, regex=True)
        df[f'{column}_binario'] = df[f'{column}_binario'].map({True: 1, False: 0})


def meaning(df):
    check_and_create_columns(df, ["Meaning_gpt3", "Meaning_davinci"])

    for index in range(len(df)):
        print(f"{index} of {len(df)}")
        word = df.loc[index, 'Word']
        df.loc[index, 'Meaning_davinci'] = asyncio.run(openai_api.ask_completion(word, type=5, timeout=120))
        df.loc[index, 'Meaning_gpt3'] = asyncio.run(openai_api.send_message_openai(word, type=5, timeout=120))

    return 


def run_experiment(words, model="ChatGPT", version="3.5", temperature="0"):
    openai_api.authentication()

    dataset_df = dataset.read_words(words)
    dataset_df = asyncio.run(loop_through(dataset_df))
    fix_empty(dataset_df)
    binary(dataset_df)
    meaning(dataset_df)
