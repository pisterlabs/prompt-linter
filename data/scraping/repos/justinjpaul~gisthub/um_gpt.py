"""Correct um gpt accessor"""

import os
from openai import AzureOpenAI

SPLIT_KEY = "****END FIRST OUTPUT****"
STARTING_PROMPT = f"""
    I am sending you two sets of notes that were taken on the same topic. 
    I want you to combine them and give me an output as a markdown file. 
    This output should capture the main idea of the notes and merge important details into one concise
    set of notes. If the two notes differ drastically in topic, always prefer the first one I send.
    Begin with a 3-4 sentence summary, then continue into details. 
    """


def read_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Prepends system prompt
# Formats user prompts
def prepare_combine_query(messages):
    return [
        {"role": "system", "content": STARTING_PROMPT},
        *[{"role": "user", "content": x} for x in messages],
    ]


def query_gpt(messages):
    client = AzureOpenAI(
        azure_endpoint="https://api.umgpt.umich.edu/azure-openai-api/ptu",
        api_key=os.getenv("UM_GPT_API_KEY"),
        api_version="2023-03-15-preview",
    )

    responses = client.chat.completions.create(model="gpt-4", messages=messages)

    return responses


# Singular response to a message
def read_gpt_multi_output(response):
    # Assumes that only ones with multioutput exist
    try:
        summary = response.message.content.strip()
        return summary

    except Exception as e:
        raise e


def create_file(filename, data):
    with open(filename, "w+") as f:
        f.write(data)


def main():
    f1 = "eecs281_1.txt"

    f_text = read_file(f1)
    sample2 = "the big o complexity of quicksort is nlogn. the big o complexity of radix sort is o(n)"

    messages = prepare_combine_query([f_text, sample2])
    response = query_gpt(messages)
    print("post query")

    x = response.choices[0]
    summary = read_gpt_multi_output(x)

    create_file("summary.md", summary)


if __name__ == "__main__":
    main()
