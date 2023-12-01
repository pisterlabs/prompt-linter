import openai
import yaml
from src import utils
import time
from pprint import pprint
import os
import argparse
from tqdm import tqdm
import re


def get_trans_msg(context):
    """
    Message for chat api
    """

    msg = [
        {
            "role": "system",
            "content": "Translate given sentence into Korean."
        },
        {
            "role": "user",
            "content": context
        }
    ]

    return msg

def generate_answer(message, model_name, max_token, temperature):
    """
    Generate response
    """
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_token,
            n=1)
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(message, model_name, max_token, temperature)

    return completion


def remove_line_split(context):
    context = re.sub(r'\n{1,}', ' ', context)
    context = re.sub(r'\s{2,}', ' ', context)
    return context



def main(args):
    api = args.api
    api_key_file = args.api_key_file

    model_name = args.model_name
    max_token = args.max_token
    temperature = args.temperature

    input_file = args.input_file
    output_path = args.output_path

    # Adding openai key
    key_dict = utils.get_keys(api_key_file)
    openai.api_key = key_dict[api]

    # Create output_file & initializing file writer
    output_file_name = 'kor-' + input_file.split('/')[-1]
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + '/' + output_file_name
    print(f'output_file : {output_file}')
    jsonl_writer = utils.JSONLWriter(output_file)

    # Reading input_file
    data = utils.read_json_file(input_file)
    data = data['data']

    for ins in tqdm(data.values()):
        instance = ins[0]

        eng_comp = remove_line_split(instance['reasoning_completion'])
        eng_query = instance['question']

        query_msg = get_trans_msg(eng_query)
        ans_msg = get_trans_msg(eng_comp)

        query_completion = generate_answer(query_msg, model_name, max_token, temperature)
        q_response = query_completion["choices"][0]["message"]["content"]

        ans_completion = generate_answer(ans_msg, model_name, max_token, temperature)
        ans_response = ans_completion["choices"][0]["message"]["content"]

        instance['kor_question'] = q_response
        instance['kor_answer'] = ans_response

        jsonl_writer.write_json_line(instance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval script for GSM8K")

    # API information
    parser.add_argument("--api", type=str, default="openai", help="API name")
    parser.add_argument("--api_key_file", type=str, default='/userhomes/philhoon/self-debate/api.yaml',
                        help="Path to the API key file")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0301", help="The model name")
    parser.add_argument("--max_token", type=int, default=2048, help="max_token_length")
    parser.add_argument("--temperature", type=float, default=0, help="model temperature")

    # Input information
    parser.add_argument("--input_file", type=str, default='/projects/self-debate/data/gsm8k/test.jsonl', help="Path to the input file")

    # Output location
    parser.add_argument("--output_path", type=str, default='/userhomes/philhoon/self-debate/result', help="Path to the output directory")

    args = parser.parse_args()
    main(args)

# python openai_translate_reasoning_api.py \
# --api openai \
# --api_key_file /userhomes/philhoon/kt-ai-challenge/api-openai.yaml \
# --model_name gpt-3.5-turbo-0613 \
# --max_token 2048 \
# --temperature 0 \
# --input_file /userhomes/philhoon/kt-ai-challenge/eng_data/D_multiarith.json \
# --output_path /userhomes/philhoon/kt-ai-challenge/eng_data
