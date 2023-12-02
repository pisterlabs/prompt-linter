#!/usr/bin/env python # pylint: disable=missing-module-docstring
# -*- coding: utf-8 -*-
# ***************************************************************************80*************************************120
#
# python ./summarize/summarize_transcripts_generic.py                                    # pylint: disable=invalid-name
#
# text mode received limited testing
#
# *********************************************************************************************************************

# standard imports
import json
import os
import logging
import logging.config
from multiprocessing import Pool
from time import perf_counter
import datetime
import re

# 3rd party imports
import openai
import pandas
import numpy
import click
import tiktoken
import humanfirst

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

@click.command()
@click.option('-i', '--input_filepath', type=str, required=True,
              help='Path containing HF Unlabelled conversations in json format or a txt format if utterances')
@click.option('-a', '--openai_api_key', type=str, required=True, help='OpenAI API key')
@click.option('-p', '--prompt', type=str, default='./prompts/abcd_example_prompt.txt',
              help='location of prompt file to read')
@click.option('-t', '--output_tokens', type=int, default=500, help='Tokens to reserve for output')
@click.option('-n', '--num_cores', type=int, default=2, help='Number of cores for parallelisation')
@click.option('-s', '--sample_size', type=int, default=0, help='Number of conversations/utterances to sample')
@click.option('-m', '--model_override', default='', type=str, required=False,
              help='Use this model name')
@click.option('-l', '--log_file_path', type=str, default='./logs', help='Server log file path')
@click.option('-b', '--drop_list', type=str, default='', help='Comma separated list of conversations to drop')
@click.option('-o', '--output_file_path', type=str, default='./summaries', help='Summaries output file path')
@click.option('-r', '--rewrite', is_flag=True, type=bool, default=False,
              help='If present will rewrite (overwrite) all previous summaries')
@click.option('-d', '--dummy', is_flag=True, type=bool, default=False, help='Skip the actual openai call')
@click.option('-v', '--verbose', is_flag=True, type=bool, default=False,
              help='Set logging level to DEBUG otherwise INFO')
def main(input_filepath: str,
         openai_api_key: str,
         num_cores: int,
         prompt: str,
         output_tokens: int,
         sample_size: str,
         model_override: str,
         log_file_path: str,
         drop_list: str,
         output_file_path: str,
         rewrite: bool,
         dummy: bool,
         verbose: bool) -> None:
    '''Main Function'''
    process(input_filepath, openai_api_key, num_cores, prompt, output_tokens,
            sample_size, model_override, log_file_path, drop_list, output_file_path,
            rewrite, dummy, verbose)


def process(input_filepath: str,
            openai_api_key: str,
            num_cores: int,
            prompt: str,
            output_tokens: int,
            sample_size: str,
            model_override: str,
            log_file_path: str,
            drop_list: str,
            output_file_path: str,
            rewrite: bool,
            dummy: bool,
            verbose: bool):
    '''Summarization of Conversations'''

    # set log level
    log_level = "INFO"
    if verbose:
        log_level = "DEBUG"

   # logging config
    now = datetime.datetime.now().isoformat()
    log_file_path = os.path.join(DIR_PATH,log_file_path,f'summarize_transcripts_generic_{now}.log')

    # locate where we are
    here = os.path.abspath(os.path.dirname(__file__))
    path_to_log_config_file = os.path.join(here,'config','logging.conf')

    # Load logging configuration
    logging.config.fileConfig(
        path_to_log_config_file,
        defaults={
            'log_file': log_file_path,
            'log_level': log_level.upper()
        }
    )

    # create logger
    logger = logging.getLogger('humanfirst.summarize')

    logger.info("Logging to: %s", log_file_path)

    # get prompt - could be local, could be run from root or here, could be absolute
    prompt_paths = [os.path.join(DIR_PATH,prompt),prompt]
    prompt_found = False
    for prompt_path in prompt_paths:
        if os.path.isfile(prompt_path):
            prompt_found = True
            prompt = open(prompt_path, mode="r", encoding="utf8").read()
            break
    assert prompt_found
    logging.info("Prompt is: \n %s", prompt)

    openai.api_key = openai_api_key

    # load input data
    if input_filepath.endswith(".json"):
        with open(input_filepath, mode="r", encoding="utf8") as file:
            data = json.load(file)
        df = pandas.json_normalize(data=data["examples"], sep="-",)

        # enforce id is string
        df["context-context_id"] = df["context-context_id"].astype(str)

        # give a sequence number to each utterance
        df = df.sort_values(["context-context_id", "created_at"])
        df['seq'] = df.groupby("context-context_id").cumcount()

        mode = "conversation"

        # set context-context_id and seq as index
        df.set_index(["context-context_id", "seq"], drop=False, inplace=True)

    elif input_filepath.endswith(".txt"):
        with open(input_filepath, mode="r", encoding="utf8") as file:
            data = file.read()
            data = data.split("\n")
        df = pandas.DataFrame(data=data, columns=["text"],)

        df["context-context_id"] = df["text"]
        df["seq"] = 1
        mode = "text"
        df.set_index(["context-context_id", "seq"], drop=False, inplace=True)
    else:
        raise RuntimeError(f"Unrecognised type: {input_filepath}")

    # work out what's been run before
    output_file_path = os.path.join(DIR_PATH,output_file_path)
    print(f'Checking: {output_file_path} for previously run')

    completed_df = get_completed_files(output_file_path)
    df = df.join(completed_df)
    df["completed"] = df["completed"].fillna(False)

    # default to run everything
    df["skip"] = False

    # skip if rewrite false
    if not rewrite:
        df["skip"] = df["completed"]

    # get all the context-context_ids
    context_ids = set(df.index.unique(level=0))
    completed_context_ids = set(completed_df.index.unique(level=0))

    # skip don't resample completed
    if not rewrite:
        context_ids = context_ids - completed_context_ids
    context_ids = list(context_ids)

    # implement dummy run flag
    if dummy:
        df["skip"] = dummy

    # take the next sample size to process
    if sample_size > 0:
        context_ids = context_ids[0:sample_size]
    logger.info(
        "In this run are going to process this many context_ids: %s", len(context_ids))
    logger.info("First up to 10: %s", context_ids[0:10])

    # select down the data frame to that.
    df = df.loc[context_ids, :]

    if mode == "conversation":
        # ABCD example has a difference between metdata-abcd_role and context-role
        # here we are going to use a generic client|expert HF standard but that is perhaps not optimum
        # print(df.loc["1000",["text","metadata-abcd_role","context-role"]])
        # build the conversation line for prompt
        df["prompt_line"] = df["context-role"] + ": " + df["text"]

        # reduce our data frame just to the data we need
        df = df[["prompt_line", "skip", "completed"]]

    elif mode == "text":
        df["prompt_line"] = df["text"]
        # reduce our data frame just to the data we need
        df = df[["context-context_id","prompt_line", "skip", "completed"]]

    else:
        raise RuntimeError(f"Not recognised mode: {mode}")

    assert isinstance(df, pandas.DataFrame)

    if mode == "conversation":
        # join all the prompt_lines together into the conversation text by
        # the context-context_id, skip and whether completed
        df = df.groupby(["context-context_id", "skip", "completed"]
                        )['prompt_line'].apply('\n'.join).reset_index()
        df.set_index("context-context_id", inplace=True, drop=False)
        df.rename(columns={"prompt_line": "conversation"}, inplace=True)

        # assemble the final prompt with the {{ conversation }} replaced
        hf_nlg = humanfirst.nlg.HFNLG("conversation")
        re_conversation = hf_nlg.get_nlg_tag_regex()

        # drops off any conversations from processing
        if drop_list != "":
            drop_list = drop_list.split(",")
            df.drop(labels=drop_list,inplace=True)

        # assemble the final prompt with the {{ conversation }} replaced
        df['prompt'] = df['conversation'].apply(
            merge_prompt_and_string, args=[prompt, re_conversation])
    elif mode == "text":
        # assemble the final prompt with the {{ conversation }} replaced
        hf_nlg = humanfirst.nlg.HFNLG("text")
        re_text = hf_nlg.get_nlg_tag_regex()

        # drops off any conversations from processing
        if drop_list != "":
            drop_list = drop_list.split(",")
            df.drop(labels=drop_list,inplace=True)

        df["prompt"] = df["prompt_line"].apply(
            merge_prompt_and_string, args=[prompt, re_text])
        df.set_index("context-context_id", inplace=True, drop=False)
    else:
        raise RuntimeError(f"Not recognised mode: {mode}")

    # estimate the tokens - use the 4k one, and then bump up if needed with factor of safety
    df['tokens'] = df["prompt"].apply(count_tokens,
                                      args=[tiktoken.encoding_for_model("gpt-3.5-turbo")])
    mean_input = df["tokens"].mean()
    logger.info('Mean input tokens is: %s', mean_input)
    logger.info('Output tokens is:  %s', output_tokens)
    in_and_out_tokens = mean_input + output_tokens
    per_second = 1500 / in_and_out_tokens
    logger.info('Per second rate is max %2f', per_second)

    # work out model for each row
    if model_override != '':
        df['model'] = model_override
    else:
        # Below line send the tokens as a series.
        # This helps in providing the index while raising OpenAITooManyTokens exception
        df['model'] = df[['tokens']].apply(calculate_which_model,args=[output_tokens],axis=1)

    # max_tokens
    df['max_tokens'] = output_tokens

    print(df[["context-context_id","prompt"]])

    # add the output location for the file
    df['summary_path'] = output_file_path

    # verbose setting
    df["verbose"] = verbose

    print(df)

    # parallelization
    pool = Pool(num_cores)
    dfs = numpy.array_split(df, num_cores)
    pool_results = pool.map(parallelise_calls, [(df, logger) for df in dfs])
    pool.close()
    pool.join()
    df = pandas.concat(pool_results, axis=1)


def get_completed_files(output_file_path: str) -> pandas.DataFrame:
    '''Create a dataframe of ids that have already been created'''
    file_names = os.listdir(output_file_path)
    completed_ids = []
    for file_name in file_names:
        if file_name.endswith(".txt"):
            completed_ids.append(file_name[0:-4])
    completed_df = pandas.DataFrame(
        completed_ids, columns=['context-context_id'])
    completed_df['completed'] = True
    completed_df.set_index(['context-context_id'], inplace=True, drop=True)
    return completed_df


def merge_prompt_and_string(text: str, prompt: str, re_tag: re) -> str:
    ''' Replaces tags with the actual string
    without using a substitution'''
    match = re_tag.search(prompt)
    if not match:
        return prompt
    assert isinstance(match, re.Match)
    limit_before_match = match.span()[0]
    start_after_match  = match.span()[1]
    prompt = prompt[0:limit_before_match] + text + prompt[start_after_match:]
    return prompt


def parallelise_calls(args) -> pandas.DataFrame:
    '''Parallelise dataframe processing'''
    df,logger = args
    return df.apply(call_api, axis=1, args=[logger])


def call_api(row: pandas.Series, logger: logging) -> pandas.Series:
    '''Call OpenAI API for summarization'''

    start_time = perf_counter()
    logger.info(
        "Calling OpenAI model %s to summarize conversation: %s", row["model"], row.name)
    if row["verbose"]:
        logger.info("Prompt: %s", row["prompt"])

    row["summary"] = ''
    row["total_tokens"] = 0
    if not row["skip"]:
        try:
            row = summarize(row, output_tokens=row["max_tokens"])
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Exception for %s is %s', row.name, e)
            return row["summary"]

    logger.info('Total tokens for conversation id %s is %s',
                 row.name, row["total_tokens"])
    logger.info('Conversation - %s is summarized', row.name)

    # write the file to output if not in dummy mode
    if not row["skip"]:
        with open(os.path.join(row["summary_path"],f'{row["context-context_id"]}.txt'),
                  mode="w", encoding="utf8") as file:
            file.write(row["summary"])

    logger.info('Summary is saved at: %s', row["summary_path"])
    end_time = perf_counter()
    logger.info('Took %.2f seconds', end_time-start_time)
    return row


def calculate_which_model(row: pandas.Series, output_tokens: int) -> str:
    '''Works out which model for data line based on tokens'''

    tokens=row["tokens"]
    # dec to bin conversion gives margin of error
    if tokens + output_tokens < 4000:
        return "gpt-3.5-turbo"
    elif tokens + output_tokens < 16000:
        return "gpt-3.5-turbo-16k"
    elif tokens + output_tokens < 32000:
        return "gpt-4-32k"
    else:
        error_string = f'Data tokens {tokens + output_tokens}'
        error_string = f'{error_string} exceeds OpenAI model limit 32000 - convo id is {row.name}'
        raise RuntimeError(error_string)


def summarize(row: pandas.Series, output_tokens: int) -> str:
    '''Summarizes single conversation using prompt'''

    response = openai.ChatCompletion.create(
        model=row["model"],
        messages=[
            {"role": "user", "content": row["prompt"]}
        ],
        temperature=0.0,
        max_tokens=output_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    row["summary"] = response.choices[0].message.content + "\n"
    row["total_tokens"] = response.usage.total_tokens
    return row


def count_tokens(text: str, encoding):
    """Returns the number of tokens in a text string."""
    return len(encoding.encode(text))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
