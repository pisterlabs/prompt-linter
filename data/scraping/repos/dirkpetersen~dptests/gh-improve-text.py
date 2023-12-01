#! /usr/bin/env python3

import openai
import requests
import os
import argparse

def read_api_key(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()

openai_api_key_path = os.path.expanduser("~/.config/openai/openai_api_key")
openai.api_key = read_api_key(openai_api_key_path)

def fetch_github_file(url):
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Error fetching file: {response.status_code}")

def improve_text(text):
    prompt = f"Please improve the following text by fixing grammar, spelling, and style:\n\n{text}"

    print('text to improve:', text)

    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": "Who won the world series in 2020?"},
        #         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #         {"role": "user", "content": "Where was it played?"}
        #     ]
        # )
        messages=[
                {"role": "user", "content": prompt}
            ]
    )    
    return response.choices[0]['message']['content'].text.strip()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Improve the text of a file from a GitHub repository.')
    parser.add_argument('arg1', help='GitHub user, repository, or full URL to raw.githubusercontent.com')
    parser.add_argument('arg2', nargs='?', help='GitHub repository (if the first argument is a user)')
    parser.add_argument('arg3', nargs='?', help='File in the GitHub repository (if the first two arguments are user and repository)')

    return parser.parse_args()

def save_to_file(content, filename):
    with open(filename, 'w') as f:
        f.write(content)

def main():
    args = parse_arguments()

    if args.arg2 is None:
        url = args.arg1
        local_filename = os.path.basename(url).split('.')[0]
        file_extension = os.path.splitext(url)[-1]
    elif args.arg3 is None:
        url = f"https://raw.githubusercontent.com/{args.arg1}/{args.arg2}/main/README.md"
        local_filename = 'README'
        file_extension = '.md'
    else:
        url = f"https://raw.githubusercontent.com/{args.arg1}/{args.arg2}/main/{args.arg3}"
        local_filename = os.path.basename(url).split('.')[0]
        file_extension = os.path.splitext(url)[-1]

    try:
        file_text = fetch_github_file(url)
        improved_text = improve_text(file_text)
        improved_filename = f"{local_filename}-improved{file_extension}"

        print(f"Original content of {local_filename}{file_extension}:")
        print(file_text)
        print(f"\nImproved content of {improved_filename}:")
        print(improved_text)

        save_to_file(improved_text, improved_filename)
        print(f"\nImproved content has been saved to {improved_filename}")

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()

