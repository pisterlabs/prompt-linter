#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***************************************************************************80
#
# python ./summarize/summarize_transcripts.py
#
# This script summarizes transcripts/conversations within
# 4k token context window (prompt + completion).
#
# This uses gpt-3.5-turbo model.
#
# Accepts
#  - Conversations/Transcripts in HumanFirst Dataformat
#  - Openai API Key
#  - Number of cores to use for parallelization (optional)
#  - Number fo conversations to summarize (optional)
#  - File path of server log (optional)
#
# Prompt format:
# """
# <contents of prompt1.txt file>
#
# <conversation/transcript>
#
# <contents of prompt2.txt file>
# """
#
# Parallelization of API calls helps to summarize large number of transcripts
#
# Saves the summary of each transcript as an individual text file
#
# *****************************************************************************

# standard imports
import json
from os.path import exists
import random
import logging
from os.path import join
from pathlib import Path
from multiprocessing import Pool
import time
from time import perf_counter

# 3rd party imports
import openai
import pandas
import numpy
import click

START_TIME = perf_counter()


@click.command()
@click.option('-i', '--input_filepath', type=str, required=True, help='Path containing HF Unlabelled conversations in json format')
@click.option('-a', '--openai_api_key', type=str, required=True, help='OpenAI API key')
@click.option('-n', '--num_cores', type=int, default=8, help='Number of cores for parallelisation')
@click.option('-c', '--conversation_count', type=int, default=100, help='Number of conversations to process')
@click.option('-s', '--server', type=str, default='', help='Server log file path')
def main(input_filepath: str, openai_api_key: str, num_cores: int, conversation_count: str, server: str):
    '''Main Function'''
    process(input_filepath, openai_api_key, num_cores, conversation_count, server)


def process(input_filepath: str, openai_api_key: str, num_cores: int, conversation_count: str, server: str):
    '''Summarization of Conversations'''

    # logging config
    if server == "":
        server = join(str(Path(input_filepath).parent), "server_summarize.log")

    logging.basicConfig(filename=server, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(process)d - %(levelname)s -- %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    openai.api_key = openai_api_key

    # load input data
    with open(input_filepath, mode="r", encoding="utf8") as f:
        data = json.load(f)
    df = pandas.json_normalize(data=data["examples"], sep="-")

    # set index as conversation ID
    df.set_index(["context-context_id"], drop=True, inplace=True)

    # randomly generate samples
    samplings_ids = list(set(df.index))
    if len(samplings_ids) > conversation_count:
        sampling_ids = random.sample(samplings_ids, conversation_count)
        df = df[df.index.isin(sampling_ids)]
    assert isinstance(df, pandas.DataFrame)

    # make one conversations per row so that when splitting,
    # all the converation utterances stays intact
    cols = list(df.columns)
    agg_dict = {}

    for col in cols:
        agg_dict.update({
            col: lambda x: x.tolist()
        })

    df = df.groupby(df.index).agg(agg_dict)

    dfs = numpy.array_split(df, num_cores)
    parallelization_input = []
    for dataframe in dfs:
        dataframe = dataframe.explode(cols)
        parallelization_input.append([dataframe, input_filepath])

    # parallelization
    with Pool(num_cores) as p:
        parallelization_output = p.map(summarization, parallelization_input)

    large_convo_id = []
    for output in parallelization_output:
        large_convo_id.extend(output)

    large_convo_id = list(set(large_convo_id))
    large_convo_path = join(str(Path(input_filepath).parent), "large_convo_id.txt")
    with open(large_convo_path, mode="w", encoding="utf8") as f:
        f.write("\n".join(large_convo_id))

    convo_processed_num = len(df.index.unique().to_list())
    large_convo_num = len(large_convo_id)
    logging.info(f"Total number of conversations processed is {convo_processed_num}")
    logging.info(f"Total number of long conversation is {large_convo_num}")
    logging.info(f"Total number of conversations summarized is {convo_processed_num - large_convo_num}")
    logging.info(f'Total Duration for the script: {time.strftime("%H:%M:%S", time.gmtime(perf_counter() - START_TIME))}')


def summarization(input: list) -> None:
    '''Summarization'''

    df = input[0]
    input_filepath = input[1]

    indices = df.index.unique().to_list()
    len_of_indices = len(indices)
    i = 0

    # reading large convo id file
    large_convo_path = join(str(Path(input_filepath).parent), "large_convo_id.txt")
    large_convo_id = []
    if exists(large_convo_path):
        with open(large_convo_path, mode="r", encoding="utf8") as f:
            large_convo_id = f.read()
            large_convo_id = large_convo_id.split("\n")
            large_convo_id = [d.strip() for d in large_convo_id]

    while i < len_of_indices:

        index = indices[i]

        # skipping the transcripts with tokens exceeding 4097
        if index in large_convo_id:
            i = i + 1
            continue

        example_df = df[df.index.isin([index])]

        # get the conversation as client-expert dialogue
        conversation = get_conversation(example_df)

        summary_path = f"{input_filepath.split('.json')[0]}_conversation_{index}_summary.txt"
        try:
            if not exists(summary_path):
                logging.warning(f"Summary for conversation ID {index} doesn't exists in the path {summary_path}")
                summary = call_api(index, conversation, summary_path)

            else:
                logging.info(f"Summary for conversation ID {index} already exists\nReading the summary from file {summary_path}")
                with open(summary_path, mode="r", encoding="utf8") as f:
                    summary = f.read()
                    if summary == "":
                        logging.warning(f"Summary file {summary_path} is empty for conversation ID {index}")
                        summary = call_api(index, conversation, summary_path)

            i = i + 1
        except Exception as e:
            logging.error(f"Error upon calling API - {e} - id - {index}")
            if f"{e}".find("4097") != -1:
                large_convo_id.append(index)
                print(f"Large Conversation ID: {index}")
                logging.warning(f"Large Conversation ID: {index}")
                i = i + 1
                continue
            else:
                sec = 5
                time.sleep(sec)
                logging.info(f"Retrying API call for conversation - {index}")
                continue

    return large_convo_id


def call_api(index: str, conversation: str, summary_path: str) -> str:
    '''Call OpenAI API for summarization'''

    logging.info(f"Calling OpenAI to summarize conversation - {index}")
    summary, total_tokens = summarize(conversation)
    logging.info(f"Total tokens for conversation id {index} is {total_tokens}")
    logging.info(f"Conversation - {index} is summarized")
    with open(summary_path, mode="w", encoding="utf8") as f:
        f.write(summary)
    logging.info(f"Summary is saved at {summary_path}")
    # print()
    # print(index)
    # print(conversation)
    return summary


def summarize(text) -> str:
    '''Summarizes single conversation using prompt'''

    with open("./summarize/prompt1.txt", mode="r", encoding="utf8") as f:
        prompt1 = f.read()

    with open("./summarize/prompt2.txt", mode="r", encoding="utf8") as f:
        prompt2 = f.read()

    prompt = f"""{prompt1}\n\n```\n{text}\n```\n\n{prompt2}"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1000,
        top_p=1,                # default value
        frequency_penalty=0.0,  # default value
        presence_penalty=0.0    # default value
    )

    return response.choices[0].message.content, response.usage.total_tokens


def get_conversation(example_df: pandas.DataFrame) -> str:
    '''Converts the conversations in HF format to customer-agent dialogue'''

    utterances = []
    for key, row in example_df.iterrows():
        if row["context-role"] == "client":
            utterances.append(f'Customer: {row["text"]}')
        else:
            utterances.append(f'Agent: {row["text"]}')
    return "\n".join(utterances)


if __name__ == "__main__":
    main()
