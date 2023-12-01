#!/usr/bin/env python # pylint: disable=missing-module-docstring
# -*- coding: utf-8 -*-
# ***************************************************************************80*************************************120
#
# python ./adversarial_supervision\
#         /scripts\
#         /initial_outbound_sup\
#         /2_evaluate_initial_outbound_supervision_prompt\
#         /2_2_redacted.py                                                               # pylint: disable=invalid-name
#
# *********************************************************************************************************************

# standard imports
from multiprocessing import Pool
import re
from  os.path import join

# 3rd party imports
import openai
import pandas
import numpy
import click
from sklearn.metrics import accuracy_score, confusion_matrix


@click.command()
@click.option('-r', '--results', type=str, default="./adversarial_supervision/results/",
              help='folders path having CSV files containing results of adversarial prompt')
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

    result_path = join(results,"adversarial_base_prompt_results.csv")

    df = pandas.read_csv(result_path,sep=",",encoding="utf8")

    df["max_tokens"] = tokens

    path = "./adversarial_supervision/prompt/adversarial_base_prompt.txt"
    with open(path, mode="r",encoding="utf8") as f:
        base_prompt = f.read()
        base_prompt = base_prompt.replace(r"{{text}}","REDACTED")

    df["customer_utterance_redacted"] = base_prompt

    df["merge_prompt_completion"] = df.apply(merge_prompt_completion,axis=1)

    prompt_path = join(prompt,"adversarial_supervision_prompt2.txt") # change this line to use 5th prompt

    with open(prompt_path, mode="r",encoding="utf8") as f:
        prompt_text = f.read()

    print(f"Prompt: \n{prompt_text}")
    print()

    df["adversarial_supervision_prompt"] = df["merge_prompt_completion"].apply(
                                            create_adversarial_supervision_prompt, args=[prompt_text])
    df["model_name"] = model

    # parallelization
    pool = Pool(num_cores)
    dfs = numpy.array_split(df, num_cores)
    pool_results = pool.map(parallelise_calls, dfs)
    pool.close()
    pool.join()
    df = pandas.concat(pool_results)

    # enforce column is string
    df["whether_offensive"] = df["whether_offensive"].astype(str)

    output_file_name = prompt_path.split("/")[-1]
    output_file_name = output_file_name.replace(".txt",f"_{model}_results.csv")
    output=join(results,output_file_name)
    with pandas.option_context('display.max_colwidth', 150,):
        print(df[["completion","whether_offensive"]])
    df.to_csv(output, sep=",", encoding="utf8", index=False)

    # calc accuracy
    list_truth = list(df["inappropriate"])
    list_pred = list(df["whether_offensive"])

    print(f"Outbound Supervision Accuracy - {model}:")
    print(f"Accuracy: {round(accuracy_score(list_truth,list_pred)*100,2)} %")
    tn, fp, fn, tp = confusion_matrix(list_truth, list_pred,labels=["no","yes"]).ravel()
    print(f"True Positive: {tp}")
    print(f"False Negative: {fn}")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    print(f"Recall: {round(recall*100,2)} %")
    print(f"Precision: {round(precision*100,2)} %")
    print(f"F1 Score: {round(f1*100,2)} %")


def create_adversarial_supervision_prompt(text: str, prompt: str) -> None:
    """Add prompt+completion to the adversarial supervision prompt"""

    return prompt.replace(r"{{conversation}}",text)


def merge_prompt_completion(row: pandas.Series):
    """Merges prompt and completion"""

    prompt_completion_pair = f'{row["customer_utterance_redacted"]}{row["completion"]}'
    prompt_completion_pair = prompt_completion_pair.replace("You are a vending machine chatbot\n\n","")
    return prompt_completion_pair


def parallelise_calls(df: pandas.DataFrame) -> pandas.DataFrame:
    '''Parallelise dataframe processing'''
    return df.apply(call_api, axis=1)


def call_api(row: pandas.Series) -> pandas.Series:
    '''Call OpenAI API for summarization'''

    row["whether_offensive"], row["total_tokens"] = summarize(row["adversarial_supervision_prompt"],
                                                              row["max_tokens"],
                                                              row["model_name"])

    row["whether_offensive"] = re.sub(r'^"*','',row["whether_offensive"])
    row["whether_offensive"] = re.sub(r'"*$','',row["whether_offensive"])
    row["whether_offensive"] = re.sub(r'\.*$','',row["whether_offensive"])
    row["whether_offensive"] = row["whether_offensive"].lower()
    return row


def summarize(prompt: str, tokens: int, model_name: str) -> str:
    '''Summarizes single conversation using prompt'''

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content, response.usage.total_tokens


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
