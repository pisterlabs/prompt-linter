import numpy as np
import pandas as pd
import openai
import os
from get_completion import get_completion
import argparse
import re



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="202204", help="where to data path"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="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",
        help="my own api key",
    )
    args = parser.parse_args()

    openai.api_key = args.api_key
    text_datas = pd.read_csv(f"{args.data_path}/df_{args.data_path}.csv")
    phone_nums = list(text_datas["phone_num"].astype("str"))
    texts = text_datas["original_text"]

    phone_num_ls = []
    proofread_response_ls = []
    keyword_ls = []
    texts_ls = []
    for phone_num, text in list(zip(phone_nums[:], texts.loc[:])):
        list_of_slang = {
            "XX1": ["OOOO OOO OO OOO OO"],
            "XX2": ["OOOO OO OOOO OOOOO O"],
            "XXX XXX": ["OOO OO OOOO"],
            "XX XX XXXX": ["OOO OO OOOO"],
            "XX XX1": ["OO OOOO OOO"],
            "XX XX2": ["OO OOOO OOO"],
            "XXX XX XX": ["OO OOOO OOO"],
        }

        prompt = f"""
        너가 해야할 것은 아래의 두 사람의 통화내용을 듣고 다음의 두가지 임무를 수행하는 것이야.
        임무 1: 통화 내용 교정하기
        임무 2: 중요한 단어 5개 추출하기

        그리고 다음의 몇가지 주의 사항을 따라.
        주의 사항 : XX

        은어 목록: ```{list_of_slang}```\

        통화 내용: ```{text}```\
        그리고 대답의 양식은 다음과 같은 방식을 따라.
        -교정된 대화 내용:
        -중요한 단어:
        """
        try:
            response = get_completion(prompt)
            pattern = r"(교정된 대화 내용:|중요한 단어:)"
            split_response = re.split(pattern, response)
            proofread_response = split_response[2]
        except openai.error.InvalidRequestError as e:
            print(f"Error occurred for text {text}: {e}")
            continue

        phone_num_ls.append(phone_num)
        texts_ls.append(text)
        proofread_response_ls.append(proofread_response)
        print("Done!")
    variables1_df = pd.DataFrame(phone_num_ls, columns=["variables1"])
    variables2_df = pd.DataFrame(texts_ls, columns=["variables2"])
    variables3_df = pd.DataFrame(proofread_response_ls, columns=["variables3"])
    result = pd.concat([variables1_df, variables2_df, variables3_df], axis=1)
    result.to_csv(f"proofread.csv", index=False)


if __name__ == "__main__":
    main()
