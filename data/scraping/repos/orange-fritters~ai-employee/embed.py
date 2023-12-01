import ast
import os
import json
from typing import List
import openai
import pandas as pd
import tiktoken
from preprocess.augment_qa.aug_qa import get_response
from preprocess.embedding.embed import get_embedding

import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


def convert_prompt_summarize(document: str,
                             title: str):
    content = f"""
    Service:
    {document}
    Title:
    {title}

    You will be provided with a description of welfare services. 
    Perform the following actions:

    1 - Write one short overall description of the service.
    2 - Write one short sentence about the content of the service.
    3 - Write two to three sentences about the content of the service.
    4 - Write one short sentence about the target of the service.
    5 - Write two to three sentences about the target of the service.
    6 - Extract 10 keywords from the given target of the service in English.
    7 - Give 3 synonyms for each extracted keywords of the target in English.
    8 - Extract 10 keywords from the given content about the service in English.
    9 - Give 3 synonyms for each extracted keywords about the content in English.

    Output the final result in json format.

    Final format:
    {{
        "overall_description" : <overall description>,
        "content_description" : <content description>,
        "content_long_description" : <content long description>,
        "target_description" : <target description>,
        "target_long_description" : <target long description>,
        "target_keywords" : <target keywords as list>,
        "target_synonyms" : <target synonyms as json>,
        "content_keywords" : <content keywords as list>,
        "content_synonyms" : <content synonyms as json>,
    }},
    """
    return [{"role": "user", "content": content}]


def convert_prompt_translate(title: str,
                             summary: str,
                             keywords: List[str]):
    keywords = ", ".join(keywords)
    content = f"""
    Title:
        {title}
    Summary:
        {summary}
    Keywords:
        {keywords}

    You will be provided with a description of welfare services. 
    Perform the following actions:

    1 - Translate the title into English.
    2 - Translate the summary into English.
    3 - Translate the keywords into English.
    

    Output the final result in json format.

    Final format:
    {{
        "title_eng" : <title in English>,
        "summary_eng" : <summary in English>,
        "keywords_eng" : <keywords in English>,
    }},
    """
    return [{"role": "user", "content": content}]


def generate_embedding():
    missing = {"missing": []}
    with open("data/summary/summary.json", "r") as f:
        summary = json.load(f)

    df = pd.DataFrame(columns=["index",
                               "category",
                               "filename",
                               "title",
                               "overall_description",
                               "content_description",
                               "content_long_description",
                               "target_description",
                               "target_long_description",
                               "target_synonyms",
                               "content_synonyms",
                               "keyword",
                               "summary"])
    for i, filename in enumerate(sorted(os.listdir('data/articles'))):
        with open(f"data/articles/{filename}") as f:
            document = f.read()
        title = summary[filename]["title"]
        res, price, delay, ver = get_response(convert_prompt_summarize(document, title))

        try:
            df.loc[i] = [i,
                         filename.split("_")[0],
                         filename,
                         title,
                         json.loads(res)["overall_description"],
                         json.loads(res)["content_description"],
                         json.loads(res)["content_long_description"],
                         json.loads(res)["target_description"],
                         json.loads(res)["target_long_description"],
                         json.loads(res)["target_synonyms"],
                         json.loads(res)["content_synonyms"],
                         summary[filename]["keywords"],
                         summary[filename]["summary_processed"]]
        except json.decoder.JSONDecodeError:
            print(res)
            missing["missing"].append(i)
            with open("preprocess/emed_final/missing.json", "a") as f:
                json.dump(missing, f, indent=4)
            break

        df.to_csv("preprocess/emed_final/emed_final_1.csv", index=False)

        print("Data: \n", res)
        print(f"[{i}/462] Price: {price:.2f} Time: {delay:.2f} Version: {ver} Title: {title} ")
        print()


def convert_dict_list():
    df = pd.read_csv("preprocess/emed_final/emed_final_1.csv")
    df['target_synonyms'] = df['target_synonyms'].apply(ast.literal_eval)
    df['content_synonyms'] = df['content_synonyms'].apply(ast.literal_eval)
    target_synonyms_col = []
    content_synonyms_col = []
    for i, row in df.iterrows():
        if isinstance(row['target_synonyms'], list):
            target_synonyms = row['target_synonyms']
        else:
            target_synonyms = row['target_synonyms']
            target_synonyms = [item for key, value in target_synonyms.items() for item in ([key] + value)]

        if isinstance(row['content_synonyms'], list):
            content_synonyms = row['content_synonyms']
        else:
            content_synonyms = row['content_synonyms']
            content_synonyms = [item for key, value in content_synonyms.items() for item in ([key] + value)]
        target_synonyms_col.append(target_synonyms)
        content_synonyms_col.append(content_synonyms)
    df['target_synonyms'] = target_synonyms_col
    df['content_synonyms'] = content_synonyms_col
    df.to_csv("preprocess/emed_final/emed_final_1.csv", index=False)


def translate():
    df = pd.read_csv("preprocess/emed_final/emed_final_final.csv")
    # df['title_eng'] = ""
    # df['summary_eng'] = ""
    # df['keywords_eng'] = ""

    missing = {"missing": []}
    for i, row in df.iterrows():
        title = row['title']
        summary = row['summary']
        keywords = row['keyword']
        res, price, delay, ver = get_response(convert_prompt_translate(title, summary, keywords))

        try:
            title_eng = json.loads(res)["title_eng"]
            summary_eng = json.loads(res)["summary_eng"]
            keywords_eng = ", ".join(json.loads(res)["keywords_eng"])

            df.loc[i, 'title_eng'] = title_eng
            df.loc[i, 'summary_eng'] = summary_eng
            df.loc[i, 'keywords_eng'] = keywords_eng

            print("Data: \n", res)
            print(f"[{i}/462] Price: {price:.2f} Time: {delay:.2f} Version: {ver} Title: {title} ")
            print()
            df.to_csv("preprocess/emed_final/emed_final.csv", index=False)

        except json.decoder.JSONDecodeError:
            print(res)
            missing["missing"].append(i)
            with open("preprocess/emed_final/missing.json", "a") as f:
                json.dump(missing, f, indent=4)


def translate_category(filename, out_filename):
    df = pd.read_csv(filename)
    convert = {
        "기타지원": "OtherSupport",
        "노령층지원": "ElderlySupport",
        "법률금융복지지원": "LegalFinancialWelfareSupport",
        "보건의료지원": "HealthcareSupport",
        "보훈대상자지원": "VeteransSupport",
        "생계지원": "LivelihoodSupport",
        "임신보육지원": "PregnancyChildcareSupport",
        "장애인지원": "DisabledPersonSupport",
        "청소년청년지원": "YouthSupport",
        "취업지원": "EmploymentSupport"
    }
    df['category_eng'] = df['category'].apply(lambda x: convert[x])
    df.to_csv(out_filename, index=False)


def embed():
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    df = pd.read_csv("preprocess/emed_final/final.csv")
    df['title_embed'] = ""
    for i, row in df.iterrows():
        doc = row['document']
        embed, token_count, price = get_embedding([doc], encoding)
        df.loc[i, 'title_embed'] = json.dumps(embed)
        print(f"[{i}/462] Price: {price:.2f} Token Count: {token_count} Title: {row['title']} ")
        df.to_csv("preprocess/emed_final/final.csv", index=False)


if __name__ == "__main__":
    # generate_embedding()
    # convert_dict_list()
    # translate()
    embed()
