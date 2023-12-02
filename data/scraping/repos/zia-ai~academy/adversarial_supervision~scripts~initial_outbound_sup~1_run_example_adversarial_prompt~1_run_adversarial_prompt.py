#!/usr/bin/env python # pylint: disable=missing-module-docstring
# -*- coding: utf-8 -*-
# ***************************************************************************80*************************************120
#
# python ./adversarial_supervision\
#         /scripts\
#         /initial_outbound_sup\
#         /1_run_example_adversarial_prompt\
#         /1_run_adversarial_prompt.py                                                  # pylint: disable=invalid-name
#
# *********************************************************************************************************************

# standard imports
from multiprocessing import Pool
import re
from os.path import join

# 3rd party imports
import openai
import pandas
import numpy
import click


@click.command()
@click.option('-r', '--results', type=str, default="./adversarial_supervision/results/",
              help='folder path where results are stored')
@click.option('-a', '--openai_api_key', type=str, required=True, help='OpenAI API key')
@click.option('-p', '--prompt', type=str, default="./adversarial_supervision/prompt/",
              help='folder containing adversarial prompt and list of customer utterances')
@click.option('-t', '--tokens', type=int, default=500, help='Tokens to reserve for output')
@click.option('-n', '--num_cores', type=int, default=2, help='Number of cores for parallelisation')
@click.option('-m', '--model', type=str, required=True, help='model name - gpt-3.5-turbo-0301 or gpt-3.5-turbo-0613')
def main(results: str,
         openai_api_key: str,
         num_cores: int,
         prompt: str,
         tokens: int,
         model: str) -> None:
    '''Main Function'''
    process(results, openai_api_key, num_cores, prompt, tokens, model)


def process(results: str,
            openai_api_key: str,
            num_cores: int,
            prompt: str,
            tokens: int,
            model: str) -> None:
    '''Run prompt'''

    openai.api_key = openai_api_key

    prompt_path = join(prompt,"adversarial_base_prompt.txt")
    customer_utterances_path = join(prompt,"10_manually_crafted_adversarial_examples.txt")

    with open(prompt_path, mode="r",encoding="utf8") as f:
        prompt_text = f.read()

    with open(customer_utterances_path, mode="r",encoding="utf8") as f:
        list_of_customer_utterances = f.read()
        list_of_customer_utterances = list_of_customer_utterances.split("\n")
        list_of_customer_utterances = [utterance.strip() for utterance in list_of_customer_utterances]

    print(f"Prompt: \n{prompt_text}")

    prompt_list = []

    i = 0
    while i<len(list_of_customer_utterances):
        full_prompt = prompt_text.replace(r"{{text}}",list_of_customer_utterances[i])
        prompt_list.append({
            "prompt": full_prompt, 
            "max_tokens": tokens 
        })
        i = i+1

    df = pandas.json_normalize(data=prompt_list)

    # set the model
    df["model"] = model

    with pandas.option_context('display.max_colwidth', 150,):
        print(df)

    # parallelization
    pool = Pool(num_cores)
    dfs = numpy.array_split(df, num_cores)
    pool_results = pool.map(parallelise_calls, dfs)
    pool.close()
    pool.join()
    df = pandas.concat(pool_results)

    # enforce this column is string
    df["completion"] = df["completion"].astype(str)

    output= prompt_path.split("/")[-1]
    output= output.replace(".txt", "_results.csv")
    output= join(results,output)
    print()
    with pandas.option_context('display.max_colwidth', 150,):
        print(df["completion"])
    df.to_csv(output, sep=",", encoding="utf8", index=False)

    completion = df["completion"].unique().tolist()
    output_text = output.replace(".csv",".txt")
    with open(output_text,mode="w",encoding="utf8") as f:
        f.write("\n\n".join(completion))

def parallelise_calls(df: pandas.DataFrame) -> pandas.DataFrame:
    '''Parallelise dataframe processing'''
    return df.apply(call_api, axis=1)


def call_api(row: pandas.Series) -> pandas.Series:
    '''Call OpenAI API for summarization'''

    row["completion"], row["total_tokens"] = summarize(row["prompt"], row["model"], row["max_tokens"])

    row["completion"] = re.sub(r'^"*','',row["completion"])
    row["completion"] = re.sub(r'"*$','',row["completion"])
    row["completion"] = row["completion"].strip(" ")
    return row


def summarize(prompt: str, model: str, tokens: int) -> str:
    '''Summarizes single conversation using prompt'''

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_tokens=tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content, response.usage.total_tokens


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
