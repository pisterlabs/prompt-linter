#!/usr/bin/env python # pylint: disable=missing-module-docstring
# -*- coding: utf-8 -*-
# ***************************************************************************80*************************************120
#
# python ./adversarial_supervision\
#         /scripts\
#         /4_evaluation_of_outbound_supervision_prompt\
#         /outbound_adv_supervision.py                                                   # pylint: disable=invalid-name
#
# *********************************************************************************************************************

# standard imports
from multiprocessing import Pool
import re
from datetime import datetime
import time

# 3rd party imports
import openai
import pandas
import numpy
import click
from sklearn.metrics import accuracy_score, confusion_matrix

class UnscuccessfulAPICallException(Exception):
    """This happens when an API call goes unsuccessful"""

class EmptyResponseException(Exception):
    """This happens when a response generated is empty"""

@click.command()
@click.option('-f', '--file_path', type=str, required=True,
              help='CSV containing prompt attack responses')
@click.option('-a', '--openai_api_key', type=str, required=True, help='OpenAI API key')
@click.option('-m', '--model', type=str, required=True, help='model name - gpt-3.5-turbo-0301 or gpt-3.5-turbo-0613')
@click.option('-p', '--prompt_path', type=str,
              default="./adversarial_supervision/prompt/adversarial_supervision_prompt5.txt",
              help='adversarial outbound supervision prompt')
@click.option('-s','--sample',type=int,default=4,help='n text to sample from dataset')
@click.option('-n', '--num_cores', type=int, default=2, help='Number of cores for parallelisation')
def main(file_path: str,
         openai_api_key: str,
         num_cores: int,
         prompt_path: str,
         sample: int,
         model: str) -> None:
    '''Main Function'''
    process(file_path, openai_api_key, num_cores, prompt_path, sample, model)


def process(file_path: str,
            openai_api_key: str,
            num_cores: int,
            prompt_path: str,
            sample: int,
            model: str) -> None:
    '''Run prompt'''

    openai.api_key = openai_api_key

    df = pandas.read_csv(file_path,sep=",",encoding="utf8")

    with open(prompt_path, mode="r",encoding="utf8") as f:
        outbound_prompt_text = f.read()

    print(f"Prompt: \n{outbound_prompt_text}")
    print(df.columns)

    if "whether_offensive" not in list(df.columns):

        df["adversarial_supervision_prompt"] = df[["response","prompt_text"]].apply(
                                                create_adversarial_supervision_prompt,
                                                args=[outbound_prompt_text],axis=1)
        df["outbound_model_name"] = model

        # an example for checking
        print(df.loc[0]["adversarial_supervision_prompt"])

        df.set_index("id",inplace=True, drop=True)

        # sample n number of rows from dataset
        df = df if sample >= df.shape[0] else df.sample(sample)

        # parallelization
        pool = Pool(num_cores)
        dfs = numpy.array_split(df, num_cores)
        pool_results = pool.map(parallelise_calls, dfs)
        pool.close()
        pool.join()
        df = pandas.concat(pool_results)

        # enforce column is string
        df["whether_offensive"] = df["whether_offensive"].astype(str)

        output_file_path = file_path.replace(".csv",f"_{model}_outbound_{datetime.now().isoformat()}.csv")
        with pandas.option_context('display.max_colwidth', 150,):
            print(df[["response","inappropriate","whether_offensive"]])
        df.to_csv(output_file_path, sep=",", encoding="utf8")
        print(f"Results are stored in {output_file_path}")


    # calc accuracy
    list_truth = list(df["inappropriate"])
    list_pred = list(df["whether_offensive"])

    print(f"Outbound Supervision Results - {model}:")
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


def create_adversarial_supervision_prompt(row: pandas.Series, prompt: str) -> None:
    """Add prompt+completion to the adversarial supervision prompt"""

    text = f"Customer: REDACTED\nCharlie: {row['response']}"
    return prompt.replace(r"{{conversation}}",text)


def parallelise_calls(df: pandas.DataFrame) -> pandas.DataFrame:
    '''Parallelise dataframe processing'''
    return df.apply(call_api, axis=1)


def call_api(row: pandas.Series) -> pandas.Series:
    '''Call OpenAI API for classification'''

    try:
        row["whether_offensive"], row["total_tokens"] = classify(row["adversarial_supervision_prompt"],
                                                                 row["outbound_model_name"])

        if row["whether_offensive"] == "":
            raise EmptyResponseException(f"Empty response generated for the text - {row.text}")

        row["whether_offensive"] = re.sub(r'^"*','',row["whether_offensive"])
        row["whether_offensive"] = re.sub(r'"*$','',row["whether_offensive"])
        row["whether_offensive"] = re.sub(r'\.*$','',row["whether_offensive"])
        row["whether_offensive"] = row["whether_offensive"].lower()
        if row["whether_offensive"] == "yes":
            row["whether_offensive"] = "no"
        elif row["whether_offensive"] == "no":
            row["whether_offensive"] = "yes"
    except (Exception, EmptyResponseException) as e: # pylint: disable=broad-exception-caught
        print(f"Rerunning {row.name} due to {e}")
        time.sleep(5)
        row = call_api(row) # rerun the text
    return row


def classify(prompt: str, model_name: str) -> str:
    '''Classifies a text into harmful or unharmful'''

    # print(model_name)
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content, response.usage.total_tokens


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
