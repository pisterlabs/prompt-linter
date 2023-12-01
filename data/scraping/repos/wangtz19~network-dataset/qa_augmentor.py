import openai
import logging
import pandas as pd
import time
import re
import argparse
from utils import set_proxy, test_proxy, set_openai_key
from tqdm import tqdm
import json
import random
from qa_generator import filter_qa
from concurrent.futures import ThreadPoolExecutor


tqdm.pandas()

with open("template/question_augmentation.json", "r") as f:
    template = json.load(f)


AUGMENT_PROMPT = """根据以下示例问题和答案，改写给定问题，不需要改写答案，要求改写后的问题与原问题的意思相同，且改写后的问题与给定答案匹配，但形式与原问题不同。
问题：
{question_example}
答案：
{answer_example}
改写后的问题：
{output_example}

问题：
{question}
答案：
{answer}
改写后的问题：
1.
"""

PROMPT_1_SHOT = """请将下述提问改写为更加口语化的形式，要求保持语义不变，保证提问与回复逻辑连贯，且提问形式更加符合日常口语习惯。以下是一些例子：
示例1
[提问1] {question1}
[回复1] {answer1}
[口语化提问1] {out_question1}

以下是需要改写的提问：
[提问] {question}
[回复] {answer}
[口语化提问]
"""


def aug_questions_by_chat(row, max_tokens=1000):
    example = random.choice(template)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": AUGMENT_PROMPT.format(
                        question_example=example['question'],
                        answer_example=example['answer'],
                        output_example=example['output'],
                        question=row['question'],
                        answer=row['answer'])
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(e)
        return ""


def aug_questions(df, num_workers=3):
    assert 'question' in df.columns and 'answer' in df.columns, \
        "input file must contain question and answer columns"
    assert num_workers > 0, "num_workers must be greater than 0"

    if num_workers == 1:
        df["aug_questions"] = df.progress_apply(aug_questions_by_chat, axis=1)
    else:
        # df_dict = df.to_dict(orient="records")
        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     results = executor.map(aug_questions_by_chat, df_dict)
        # df["aug_questions"] = list(results)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _, row in df.iterrows():
                futures.append(executor.submit(aug_questions_by_chat, row))
            for future in tqdm(futures):
                future.result()
            df["aug_questions"] = [future.result() for future in futures]
    
    df["aug_questions"] = "1." + df.aug_questions
    question_list, answer_list, old_question_list = [], [], []
    for idx, row in df.iterrows():
        aug_questions = row.aug_questions
        aug_questions = re.split(r"\d+\.", aug_questions)
        aug_questions = [q.strip() for q in aug_questions if q.strip()]
        question_list.extend(aug_questions)
        answer_list.extend([row.answer] * len(aug_questions))
        old_question_list.extend([row.question] * len(aug_questions))

    new_df = pd.DataFrame({
        "question": question_list,
        "old_question": old_question_list,
        "answer": answer_list
    })
    return new_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i" ,type=str, required=True, help="input file path, csv and jsonl are supported")
    parser.add_argument("--output_format", "-o", type=str, default="jsonl", help="output file format, csv or jsonl")
    parser.add_argument("--proxy", "-p", type=str, default=None, help="proxy address")
    parser.add_argument("--key_path", "-k", type=str, default=".openai-key2", help="openai key path")
    parser.add_argument("--num_workers", "-n", type=int, default=3, help="number of workers for parallel processing")
    args = parser.parse_args()

    input_format = args.input.split(".")[-1]
    assert input_format in ["csv", "jsonl"], "input file format must be csv or jsonl"
    assert args.output_format in ["csv", "jsonl"], "output file format must be csv or jsonl"
    if args.proxy is not None:
        set_proxy(proxy=args.proxy)
    else:
        set_proxy()
    assert test_proxy(), "proxy is not working"
    set_openai_key(args.key_path)
    
    if input_format == "csv":
        df = pd.read_csv(args.input)
    elif input_format == "jsonl":
        df = pd.read_json(args.input, lines=True)
    else:
        raise NotImplementedError
    if "question" not in df.columns or "answer" not in df.columns:
        assert "prompt" in df.columns and "completion" in df.columns,\
            "input file must contain (prompt and completion) columns" + \
            "or (question and answer) columns"
        df.rename(columns={"prompt": "question", "completion": "answer"}, inplace=True)
    new_df = aug_questions(df, num_workers=args.num_workers)
    tmp_path = args.input.replace(".csv", "-aug.csv")
    new_df.to_csv(tmp_path, index=False)
    filter_qa(tmp_path, output_format=args.output_format)
        

if __name__ == "__main__":
    main()