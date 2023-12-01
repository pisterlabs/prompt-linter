import os, sys
import argparse
import pandas as pd


import openai

import json


def retrieve_api_key_file(api_key_file):
    with open(api_key_file) as f:
        json_key = json.load(f)    

    return json_key["key"]



def get_response_from_llm(text):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    answer = response["choices"][0]["text"]

    if answer.strip().lower().startswith("true"):
        return True
    else:
        return False


def compose_question(col1, col2):
    text = "Q: There is a clear correlation between " + col1 + " and " + col2 + ", True or False?\n\n A:"
    return text


def output_correlated_attributes(correlated_col_pairs, output_dir):
    output_file_name = os.path.join(output_dir, "correlated_attrs")

    with open(output_file_name, "w") as f:
        for col_pair in correlated_col_pairs:
            f.write(col_pair[0] + "," + col_pair[1] + "\n")

        f.close()
    


def check_correlations_between_attributes(column_names):

    correlated_col_pairs = []

    for idx1 in range(len(column_names)):
        for idx2 in range(len(column_names)):
            if idx1 < idx2:
                col1 = column_names[idx1]
                col2 = column_names[idx2]
            
                text = compose_question(col1, col2)
                correlated_count = 0
                for k in range(6):
                    correlated = get_response_from_llm(text)
                    if correlated:
                        correlated_count += 1
                if correlated_count >= 2:
                    print("correlated attributes::", col1, col2)
                    correlated_col_pairs.append((col1, col2))

    return correlated_col_pairs





def obtain_all_attributes(data_dir):
    sample_file = os.path.join(data_dir, "train/2092/episode1_timeseries.csv")
    sample_df = pd.read_csv(sample_file)

    column_names = sample_df.columns[1:]

    return column_names


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data_dir', help='input folder')
    parser.add_argument('--api_key_file', help='input folder')
    parser.add_argument('--output', help='output folder')
    # parser.add_argument(
    #     '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    # parser.add_argument(
    #     '--show', type=bool, default=False, help='display option')
    # parser.add_argument(
    #     '--wait', type=int, default=0, help='cv2 wait time')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    api_key = retrieve_api_key_file(args.api_key_file)

    openai.api_key = api_key

    column_names = obtain_all_attributes(args.data_dir)

    correlated_col_pairs = check_correlations_between_attributes(column_names)

    output_correlated_attributes(correlated_col_pairs, args.output)

if __name__ == '__main__':
    main()
