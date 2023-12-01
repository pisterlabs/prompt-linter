import os
import openai
import time
import json

import pandas as pd
from preprocess.augment.augment import count_token, count_tokens
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


@retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3))
def get_response(text):
    start = time.time()
    tokens = count_tokens(text)

    response = openai.ChatCompletion.create(
        model=(ver := "gpt-3.5-turbo-16k"),
        messages=text,
        temperature=0
    )
    price = 0.0015 * tokens * 1.3
    price += 0.002 * count_token(response['choices'][0]['message']['content']) * 1.3
    return response['choices'][0]['message']['content'], price, time.time() - start, ver


def convert_prompt(document, title):
    content = f"""
    Service:
    {document}
    Title:
    {title}

    You will be provided with a description of welfare services. 
    Perform the following actions:

    1 - Write the one short sentence of description of the service.
    2 - Rename the title according to the description.
    3 - Convert the title to more daily and easy tone in Korean.
        * Write the reason why conveted title is more daily and easy.
    4 - Write a speicfic question that can be asked from the given document.
        * Question must include the converted title.
        * Question must be about target and contents of the service.
    5 - Translate the question into Korean using the converted title.
        * Never use the original title in Korean.

    Output the final result in json format.

    Final format:
    {{
        "description" : <description>,
        "renamed_title" : <converted title>,
        "converted_title" : <daily title>,
        "reason" : <reason>,
        "question_eng" : <question> ,
        "question_kor" : <translated question>,
    }},

    Quesion and translation must include the converted easy version title.
    """
    return [{"role": "user", "content": content}]


if __name__ == "__main__":
    data = {}
    for i, filename in enumerate(sorted(os.listdir('data/articles'))):
        if i % 3 != 0 or i < 132:
            continue
        with open(f"data/articles/{filename}") as f:
            document = f.read()

        title = document.split('<span style="font-weight: bold">')[1].split('</span>')[0].strip()
        res, price, delay, ver = get_response(convert_prompt(document, title))

        try:
            data[i] = {
                "data": json.loads(res),
                "desc": {"filename": filename,
                         "title": title,
                         "index": i,
                         "category": filename.split("_")[0],
                         "version": ver,
                         }
            }
        except json.decoder.JSONDecodeError:
            print(res)
            continue

        with open("preprocess/augment_qa/qa_data_2.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Data: \n", res)
        print(f"[{i}/462] Price: {price:.2f} Time: {delay:.2f} Version: {ver} Title: {title} ")
        print()
