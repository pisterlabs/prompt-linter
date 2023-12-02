#!/usr/bin/env python # pylint: disable=missing-module-docstring
# -*- coding: utf-8 -*-
# ***************************************************************************80*************************************120
#
# python ./adversarial_supervision\
#         /scripts\
#         /1_attack\
#         /adversarial_attack.py                                                         # pylint: disable=invalid-name
#
# *********************************************************************************************************************

# standard import
import uuid
import os
from os.path import join, exists
from datetime import datetime
from multiprocessing import Pool

# 3rd party imports
import click
import pandas
import openai
import requests
import numpy

class UnrecognisedEnvironmentException(Exception):
    """This happens when entered environmenis neither dev nor prod"""

class UnscuccessfulAPIConnectionException(Exception):
    """This happens when an API connection goes unsuccessful"""

class EmptyResponseException(Exception):
    """This happens when a response generated is empty"""

class ServerOverloadException(Exception):
    """This happens when Openai server is overloaded"""

class ResponseGeneratorMismatchException(Exception):
    """This happens when response generator is neither openai nor charlie"""

class UnscuccessfulAPICallException(Exception):
    """This happens when an API call goes unsuccessful"""

@click.command()
@click.option('-a', '--openai_api_key', type=str, required=True, help='OpenAI API key')
@click.option('-f', '--file_path', type=str,
              default='./adversarial_supervision/dataset/transfer_expriment_behaviors.csv',
              help='Input CSV file path')
@click.option('-b', '--adversarial_suffix_file_path', type=str,
              default='./adversarial_supervision/dataset/adv_suffix.txt',
              help="File path of adversarial Suffix")
@click.option('-d', '--dan_attack_file_path', type=str,
              default="./adversarial_supervision/dataset/jailbreak_dan_prefix.txt",
              help="File path of DAN attack")
@click.option('-r', '--reply_folder_path', type=str, required=True,
              help='folder where all the replies are stored')
@click.option('-u','--username' ,type=str,default="",help='username of HTTP endpoint in Node-RED')
@click.option('-p','--password' ,type=str,default="",help='password of HTTP endpoint in Node-RED')
@click.option('-s','--sample',type=int,default=4,help='n text to sample from dataset')
@click.option('-n', '--num_cores', type=int, default=2, help='Number of cores for parallelisation')
@click.option('-v', '--use_adv_suffix', is_flag=True, default=False, help='Flag for adversarial suffix usage')
@click.option('-k', '--use_dan_attack_prefix', is_flag=True, default=False, help='Flag for DAN attack usage as prefix')
@click.option('-e','--env' ,type=click.Choice(['dev', 'prod']),default='dev',help='Dev or prod to update')
@click.option('-g','--get_response_from' ,
              type=click.Choice(['openai', 'charlie']),
              default='openai',
              help='Get response from OpenAI or Charlie')
@click.option('-m', '--model', type=str, required=True, help='model name - gpt-3.5-turbo-0301 or gpt-3.5-turbo-0613')
def main(openai_api_key: str, file_path: str, adversarial_suffix_file_path: str,
         dan_attack_file_path: str, reply_folder_path: str, sample: int, num_cores: int,
         use_adv_suffix: bool, use_dan_attack_prefix: bool, get_response_from: str, env: str,
         username: str, password: str, model: str) -> None:
    '''Main Function'''

    process(openai_api_key,
            file_path,
            adversarial_suffix_file_path,
            dan_attack_file_path,
            reply_folder_path,
            sample,
            num_cores,
            use_adv_suffix,
            use_dan_attack_prefix,
            get_response_from,
            env,
            username,
            password,
            model)


def process(openai_api_key: str,
            file_path: str,
            adversarial_suffix_file_path: str,
            dan_attack_file_path: str,
            reply_folder_path: str,
            sample: int,
            num_cores,
            use_adv_suffix: bool,
            use_dan_attack_prefix: bool,
            get_response_from: str,
            env: str,
            username: str,
            password: str,
            model: str) -> None:
    '''calls_openai for the attack'''

    openai.api_key = openai_api_key

    # read dataset
    df = pandas.read_csv(file_path,encoding="utf8",header=None)
    df.rename(columns={0:"prompt_text"},inplace=True)

    # set the GUID
    # GUIDs are read from a text file if it was already present otherwise it creates them and
    # writes it into a text file. This is because it helps in comparing the results of utterances
    # in multiple runs
    guid_filepath = file_path.replace(".csv","_guid.txt")

    if exists(guid_filepath):
        with open(guid_filepath,mode="r",encoding="utf8") as f:
            guid_list = f.read()
            guid_list = guid_list.strip().split("\n")
            if df.shape[0] != len(guid_list):
                for _ in range(abs(df.shape[0] - len(guid_list))):
                    guid_list.append(f"{uuid.uuid4()}")
        df["id"] = guid_list
    else:
        df["id"] = df.apply(set_guid,axis=1)

    guid_list = df["id"].unique().tolist()
    with open(guid_filepath, mode="w", encoding="utf8") as f:
        f.write("\n".join(guid_list))

    # strip if text has any whitespaces prefixed or suffixed
    df["prompt_text"] = df["prompt_text"].apply(lambda x: x.strip())

    # reply path
    df["reply_path"] = df["id"].apply(lambda x: join(reply_folder_path,f"{x}.txt"))

    df["final_prompt"] = df['prompt_text']

    if use_dan_attack_prefix:
        # read adversarial suffix
        with open(dan_attack_file_path, mode="r", encoding="utf8") as f:
            dan_attack = f.read()

        df["dan_attack_prefix"] = dan_attack
        df["final_prompt"] = df["dan_attack_prefix"] + df["final_prompt"]

    if use_adv_suffix:
        # read adversarial suffix
        with open(adversarial_suffix_file_path, mode="r", encoding="utf8") as f:
            adv_suffix = f.read()

        df["adv_suffix"] = adv_suffix
        df["final_prompt"] = df["final_prompt"] + df["adv_suffix"]


    # get all the ids and corresponding text which are used to generate response
    completed_text_ids, completed_texts = get_completed_text_ids(reply_folder_path)
    uncompleted_text_ids = list(set(guid_list) - set(completed_text_ids))

    df.set_index("id",inplace=True, drop=True)

    uncompleted_df = df.loc[uncompleted_text_ids]
    uncompleted_df["response"] = ""
    uncompleted_df["completed"] = False

    completed_df = df.loc[completed_text_ids]
    completed_df["response"] = completed_texts
    completed_df["completed"] = True

    df = pandas.concat([completed_df,uncompleted_df])

    # set where to get the response from
    df["get_response_from"] = get_response_from

    # set the model name
    df["model"] = model

    if get_response_from == "charlie":

        # set username and password
        df["username"] = username
        df["password"] = password

        # set the url depending on which system is being used - dev | prod
        if env == 'dev':
            url = "https://elb.devvending.com/api/predict"
        elif env == 'prod':
            url = "https://elb.cwrtvending.com/api/predict"
        else:
            raise UnrecognisedEnvironmentException('Unrecognised environment')

        # set the api
        df["url"] = url


    print(df[["final_prompt","completed"]])

    # sample n number of rows from dataset
    df = df if sample >= df.shape[0] else df.sample(sample)

    print(df["prompt_text"])

    # parallelization
    pool = Pool(num_cores)
    dfs = numpy.array_split(df, num_cores)
    pool_results = pool.map(parallelise_calls, dfs)
    pool.close()
    pool.join()
    df = pandas.concat(pool_results)

    if get_response_from == "charlie":
        df.drop(columns=["username","password"],inplace=True)

    print(df[["response","get_response_from","completed"]])


    # write the final result with the timestamp
    df.to_csv(join(reply_folder_path,
                   f"final_result_{datetime.now().isoformat()}.csv"),
                   sep=",",
                   index=True,
                   encoding="utf8")
    print(f"Results are stored in {reply_folder_path}")


def set_guid(_: pandas.Series) -> str:
    """Sets the GUID for a text"""

    return str(uuid.uuid4())

def get_completed_text_ids(output_file_path: str) -> tuple:
    '''Find ids that have already been created'''

    file_names = os.listdir(output_file_path)
    completed_texts = []
    completed_ids = []
    for file_name in file_names:
        if file_name.endswith(".txt"):
            with open(join(output_file_path,file_name), mode="r", encoding="utf8") as f:
                text = f.read().strip()
                if text != "":
                    completed_ids.append(file_name[0:-4])
                    completed_texts.append(text)
    return completed_ids, completed_texts


def parallelise_calls(df: pandas.DataFrame) -> pandas.DataFrame:
    '''Parallelise dataframe processing'''

    return df.apply(send_text, axis=1)


def send_text(row: pandas.Series) -> pandas.Series:
    """Send text"""

    if row["get_response_from"] == "openai":
        return send_text_to_openai(row)
    elif row["get_response_from"] == "charlie":
        return send_text_to_charlie(row)
    else:
        raise ResponseGeneratorMismatchException(
            f"The provided response generator - {row['get_response_from']} is neither openai nor charlie")


def send_text_to_openai(row: pandas.Series) -> pandas.Series:
    '''Send text to OpenAI'''

    if not row["completed"]:

        try:
            response  = call_openai(row["final_prompt"], row["model"])
            if not response.choices[0].message.content:
                raise ServerOverloadException("Unsuccessful API Call - may be dude to server overload")

            text = response.choices[0].message.content

            text = text.strip()
            if text == "":
                raise EmptyResponseException(f"Empty response generated for the text - {row.text}")

            # Writing to text file
            with open(row["reply_path"],mode="w",encoding="utf-8") as f:
                f.write(text)
            row["response"] = text
            row["completed"] = True

        except (Exception, ServerOverloadException, EmptyResponseException) as e: # pylint: disable=broad-exception-caught
            print(f"Rerunning {row.name} due to {e}")
            row = send_text_to_openai(row) # rerun the text

    return row

def call_openai(text: str, model: str) -> str:
    '''Calling OpenAI'''

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        max_tokens=500,
        top_p=1,                # default value
        frequency_penalty=0.0,  # default value
        presence_penalty=0.0    # default value
    )

    return response

def send_text_to_charlie(row: pandas.Series) -> pandas.Series:
    """Send text to Charlie"""

    if not row["completed"]:
        data = {
            "id": row.name,
            "text": row["final_prompt"]
        }

        response  = requests.post(url=row.url, # pylint: disable=missing-timeout
                                auth=(row.username,row.password),
                                json=data)

        try:
            if response.status_code != 200:
                raise UnscuccessfulAPICallException(
                    f"Status Code :{response.status_code} \n\nResponse:\n\n{response.json()}")

            text = response.text.strip()
            if text == "":
                raise EmptyResponseException(f"Empty response generated for the text - {row.text}")

            # Writing to text file
            with open(row["reply_path"],mode="w",encoding="utf-8") as f:
                f.write(text)
            row["response"] = text
            row["completed"] = True

        except (Exception, UnscuccessfulAPICallException, EmptyResponseException) as e: # pylint: disable=broad-exception-caught
            print(f"Rerunning {row.name} due to {e}")
            row = send_text_to_charlie(row) # rerun the text

    return row

if __name__=="__main__":
    main() # pylint: disable=no-value-for-parameter
