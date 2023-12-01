"""
run:
python -m generate_data generate_instructions_from_websites
"""

import os
import re
import time

import openai
from openai import openai_object
import pandas as pd
from pathlib import Path
import json
import requests
from bs4 import BeautifulSoup
from requests.exceptions import Timeout
from typing import Union
import fire

os.environ["OPENAI_API_KEY"] = "API_KEY"

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

def get_text_from_url(url: str) -> list[list[str]]:
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        print(f"Skipping {url} due to timeout")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    text = [p.text for p in soup.find_all('p') if len(p.text) > 50]

    tokens = ' '.join(text).split()

    # split page up into x token chunks
    truncated_text = split_tokens(tokens, 850)

    return truncated_text


def split_tokens(token_list: list[str], max_tokens: int = 1000) -> list[list[str]]:
    return [token_list[i:i+max_tokens] for i in range(0, len(token_list), max_tokens)]


def encode_prompt(website: str) -> list[str]:
    prompt = open('./prompt.txt', 'r').read()
    prompts = []
    website_chunks = get_text_from_url(website)
    for website_text in website_chunks:
        prompts.append(prompt.format(website_text))
    return prompts


def get_response(
    website: str,
    temperature: float = 0.6,
    max_tokens: int = 500,
    top_p: float = 1,
    n: int = 1,
    steam: bool = False,
    frequency_penalty: float = 0.1,
    presence_penalty: float = 0,
    logit_bias={"50256": -100},  # prevent the  token from being generated
) -> list[StrOrOpenAIObject]:

    prompts = encode_prompt(website)
    if not prompts:
        return []
    start_time = time.time()  # measure the start time of the response

    responses = []
    for prompt in prompts:

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stream=steam,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias
        )
        responses.append(response)

    end_time = time.time()  # measure the end time of the response

    # print the time it took for the response for a website
    print(f" Response Time: {end_time - start_time:.2f} seconds")

    return responses


def post_process_response(response) -> str:
    if not response:
        return ""

    raw_instructions = response["choices"][0]["message"]["content"]
    QA_pairs = raw_instructions.split("\n\n")

    # remove the last QA pair if it's incomplete
    if response["choices"][0]["finish_reason"] == "length":
        QA_pairs.pop()

    return '\n\n'.join(QA_pairs)


def add_to_df(input: str, df: pd.DataFrame) -> pd.DataFrame:
    if not input:
        return df

    qna_list = input.split('\n\n')
    new_df = pd.DataFrame(
        [qna.split('\n') for qna in qna_list if len(qna.split('\n')) == 2], columns=['Questions', 'Answers']
    )

    result_df = pd.concat([df, new_df], ignore_index=True)

    return result_df


def get_raw_data() -> None:
    with open("./sources.txt", "r", encoding="utf-8-sig") as f:
        websites = [line.strip() for line in f.readlines()]

    QA_df = pd.DataFrame(columns=['Questions', 'Answers'])

    for website in websites:
        responses = get_response(website)
        if not responses:
            continue
        for response in responses:
            response_text = post_process_response(response)
            QA_df = add_to_df(response_text, QA_df)
        print(f"Total QA pairs: {QA_df.shape[0]}")

    QA_df.to_csv("Q&A_raw.csv", index=False)

def format_data() -> None:
    txt = Path('Q&A_raw.txt').read_text()
    txt = re.sub(r"\bQ:\s", "*Q*: ", txt)
    txt.replace('\n', '')
    txt = txt.split("*Q*: ")[1:]
    qlst = []
    alst = []
    for pair in txt:
        if "A: " in pair:
            ind = pair.index("A: ")
            q = pair[:ind]
            a = pair[ind + 2:]
            qlst.append(q.strip())
            alst.append(a.strip())
        else :
            continue

    qa_dict = {"instruction":qlst, "output":alst}
    df = pd.DataFrame(qa_dict)
    df["input"] = ""
    df = df[["instruction", "input", "output"]]

    df_json = pd.read_json('alpaca_data_cleaned.json')
    df_json = df_json.sample(frac=0.5)

    df_qa = pd.read_csv("Q&A_raw.csv")
    df_qa["instruction"] = ""
    df_qa["input"] = [word[2:] for word in df_qa["Questions"]]
    df_qa["output"] = [word[2:] for word in df_qa["Answers"]]
    df_qa.drop(["Questions", "Answers"],axis=1, inplace=True)

    df_final = pd.concat([df, df_json, df_qa])
    df_final.reset_index(inplace=True)

    json_list = []
    for index, row in df_final.iterrows():
        json_list.append({"instruction": row["instruction"], "input" : row["input"], "output": row["output"]})
    
    with open('final_data.json', 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=2)

def generate_instructions_from_websites() -> None:
    get_raw_data()
    format_data()
    

if __name__ == "__main__":
    fire.Fire(globals())
