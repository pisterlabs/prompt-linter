import time
from datetime import datetime

import openai
from openai.error import ServiceUnavailableError, APIError, Timeout, RateLimitError

from .clean import clean_text
from .constants import TOPICS, get_api_base_url, get_api_key


def get_tweet_count_per_label(path_to_clean_csv: str) -> dict:
    with open(path_to_clean_csv, 'r') as f:
        lines = f.readlines()[1:]

    counts = {}
    for topic in TOPICS.values():
        counts[topic] = 0

    for line in lines:
        line = line[:-1]  # remove \n
        label = line.split(',')[-1]
        counts[label] += 1

    return counts


def get_tweet_of_label(
        api_key: str,
        api_base_url: str,
        label: str,
        temperature: float,
        sleep_seconds: int = 10
) -> str:
    openai.api_key = api_key
    openai.api_base = api_base_url

    system_message = "Generate an informal Persian tweet about the given topic without any hashtags, mentions, links, or emojis."  # noqa
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {"role": "user", "content": f"topic: {label}"}
    ]

    while True:
        try:
            print("*" * 100)
            print(messages)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                timeout=120
            )
            print(response)
            print("*" * 100)

            tweet = str(response['choices'][0]['message']['content']).strip()
            tweet, _ = clean_text(tweet)
            break
        except (ServiceUnavailableError, APIError, KeyError, Timeout, RateLimitError):
            print(f"Some error occurred. Sleeping for {sleep_seconds} seconds and trying again")
            time.sleep(sleep_seconds)

    print(f"Sleeping for {sleep_seconds} seconds to avoid rate limit")
    time.sleep(sleep_seconds)
    return tweet


def append_augmented_data(path_to_csv: str, new_lines: list[str]):
    with open(path_to_csv, 'a') as f:
        for new_line in new_lines:
            f.write(new_line)


def augment_label(
        label: str, count_to_generate: int, path_to_augment_append: str, temperature: float = 0.7) -> list[str]:
    api_key = get_api_key()
    api_base_url = get_api_base_url()

    tweet_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
    tweet_owner = 'chatGPT'
    owner_university = 'openai'
    owner_name = 'openai'

    new_lines = []
    for i in range(count_to_generate):
        tweet_text = get_tweet_of_label(api_key, api_base_url, label, temperature)
        print(f"generated and cleaned tweet {i} about {label}: {tweet_text}")
        new_line = f"{tweet_time},{tweet_owner},{tweet_text},{owner_university},{owner_name},{label}\n"
        append_augmented_data(path_to_augment_append, [new_line])
        new_lines.append(new_line)
    return new_lines
