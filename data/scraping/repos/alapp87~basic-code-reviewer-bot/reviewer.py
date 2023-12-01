import os
import openai
import argparse
from dotenv import load_dotenv

SYSTEM_PROMPT = """
You will recieve a file's contents as text.
Generate a code review for the file. Indicate what changes should be made to improve its
style, performance, readability, and maintainability. If there are any reputable
libraries that could be introduced to improve the code, suggest them. Be kind and
constructive. For each suggested change, include line numbers to which you are referring.
"""


def main(file, model):
    code_review(file, model)


def code_review(filepath, model):
    with open(filepath, "r") as file:
        content = file.read()
    make_code_review_request(content, model)


def make_code_review_request(filecontent, model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Code review the following file: {filecontent}"},
    ]

    print("-------------------")

    res = openai.ChatCompletion.create(model=model, messages=messages, stream=True)

    for text in res:
        # print(text)
        delta = text["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple code reviewer for a file")
    parser.add_argument("file")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    args = parser.parse_args()

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    main(args.file, args.model)
