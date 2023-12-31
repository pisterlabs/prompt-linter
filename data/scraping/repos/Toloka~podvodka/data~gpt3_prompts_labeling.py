import openai
import pandas as pd
from openai.error import RateLimitError, ServiceUnavailableError, APIError
from tqdm import tqdm
import time
import logging
import argparse
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

PRICE_FOR_1000_TOKENS = 0.02


def prepare_gpt3_prompt(few_shot_string: str, prompt_string: str) -> str:
    return few_shot_string.format(prompt_string)


def load_logs(logs_filename: str, logger: logging.Logger) -> Tuple[int, Dict[str, str]]:
    total_tokens = 0
    prompts_to_description = dict()
    with open(logs_filename) as f:
        file_lines = f.readlines()
        if len(file_lines) == 0:
            logger.info("{} | {} | {} | {}".format('sent_text', 'respond_text', 'tokens_spent', 'total_tokens_spent'))
        else:
            for line in file_lines[1:]:
                prompt_log, description_log, tokens_spent_log, total_tokens_spent_log = \
                    ': '.join(line.split(': ')[1:]).split(' | ')
                prompts_to_description[prompt_log] = description_log
                total_tokens += int(tokens_spent_log)
    return total_tokens, prompts_to_description


def label_data(discord_prompts_iterator: iter, prompts_to_description: Dict[str, str], total_tokens: int,
               few_shot_prompt: str, logger: logging.Logger) -> Dict[str, str]:
    exception_flag = True
    while True:
        try:
            if exception_flag:
                discord_prompt = next(discord_prompts_iterator)
            if discord_prompt in prompts_to_description:
                continue
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prepare_gpt3_prompt(few_shot_prompt, discord_prompt),
                temperature=0.5,
                max_tokens=2300
            )
            image_description = response['choices'][0]['text']
            tokens = response['usage']['total_tokens']
            total_tokens += tokens
            prompts_to_description[discord_prompt] = image_description
            logger.info("{} | {} | {} | {}".format(discord_prompt, image_description, str(tokens), str(total_tokens)))
            tqdm.write(
                "discord prompt: {}\ngenerated image description: {}\ntokens spent: {}, total money spent: ${}\n" \
                .format(discord_prompt, image_description, tokens,
                        round(total_tokens / 1000 * PRICE_FOR_1000_TOKENS, 2)),
                end='\n')
            exception_flag = True
        except (RateLimitError, ServiceUnavailableError, APIError):
            exception_flag = False
            time.sleep(1.0)
            continue
        except StopIteration:
            break
    return prompts_to_description


def convert_dict_to_df(prompts_to_description: Dict[str, str], remove_prompts_without_meaning: bool) -> pd.DataFrame:
    prompts = []
    image_descriptions = []
    for prompt, image_description in prompts_to_description.items():
        prompts.append(prompt)
        image_descriptions.append(image_description)

    res_df = pd.DataFrame({'prompt': prompts, 'image_description': image_descriptions})
    if remove_prompts_without_meaning:
        res_df = res_df[res_df.image_description != '_no_object']
    return res_df


def main():
    parser = argparse.ArgumentParser(description='Generate image descriptions from prompts for Stable Diffusion')

    parser.add_argument('-t', '--token', type=str, required=True, help='API token for OpenAI')
    parser.add_argument('-org', '--organization', type=str, required=False, default='',
                        help='ID of the organization in OpenAI')
    parser.add_argument('-o', '--output', type=str, required=False, default='result.csv',
                        help='Path to the output CSV file')
    parser.add_argument('-i', '--input', type=str, required=False, default='cleaned_discord_prompts.tsv',
                        help='Path to the input TSV file generated by data_cleaning.ipynb')
    parser.add_argument('-r', '--remove_prompts_without_meaning', type=bool, required=False, default=True,
                        help='''Determine whether Discord prompts that do not describe a specific object should 
                        be removed from the output file. If not, the image description will have the value
                        "_no_object".''')

    args = parser.parse_args()
    output_path = args.output

    openai.api_key = args.token
    if args.organization:
        openai.organization = args.organization

    with open('few_shot_prompt.txt') as f:
        few_shot_prompt = f.read()

    logger = logging.getLogger(__name__)
    logs_filename = 'api_requests.log'
    file_handler = logging.FileHandler(logs_filename)
    formatter = logging.Formatter(
        '%(asctime)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    total_tokens, prompts_to_description = load_logs(logs_filename, logger)

    dataset = pd.read_csv(args.input, sep='\t')
    dataset = dataset[~dataset.content.isna()]
    dataset = dataset[~dataset.content.duplicated()].reset_index(drop=True)
    discord_prompts_iterator = iter(tqdm(dataset.content.values))
    prompts_to_description = label_data(discord_prompts_iterator, prompts_to_description, total_tokens,
                                        few_shot_prompt, logger)

    res_df = convert_dict_to_df(prompts_to_description, args.remove_prompts_without_meaning)
    res_df.to_csv(output_path, index=False)

    df = res_df[res_df.image_description != '_no_object']
    df['string'] = [f'[BOS]{x.image_description} = {x.prompt}<endoftext>' for x in df.itertuples()]
    train, test = train_test_split(df.string.values, train_size=0.8, random_state=42)

    with open('train_strings.txt', 'w') as f:
        f.write(''.join(train))

    with open('test_strings.txt', 'w') as f:
        f.write(''.join(test))


if __name__ == '__main__':
    main()
