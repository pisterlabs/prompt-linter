import os
import logging
import argparse

import openai
from pathlib import Path


SOCIALS_SYSTEM_PROMPT = '''
You are an expert social media publicist.
The user will give you a markdown file of an email newsletter.
You will write social media posts for Twitter and LinkedIn.
The posts should contain a catchy introduction, a list of the most interesting articles (especially those in the top news section), and a short call to action (to open the newsletter)
The overall tone should be fun and engaging.
The Twitter post should be within 128 characters.
The LinkedIn post should be within 500 characters.
Use creative and fun emojis and hashtags.
Respond in markdown format with Twitter and LinkedIn as section headers.
Include a link to the newsletter at the end of each post.
The newsletter link is in the "redirect" field of the markdown file.
'''


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--digest_number', '-n', type=int, required=True)
    args = parser.parse_args()

    n = args.digest_number
    digests_path = Path('../_posts/digests')
    digest_md_path = None
    for fname in os.listdir(digests_path):
        if fname.endswith(f'{n}.md'):
            digest_md_path = digests_path / fname
            break
    if digest_md_path is None:
        raise ValueError('Could not find digest md file!')
    logging.info(f'Found digest md file at {digest_md_path}')

    with open(digest_md_path, 'r') as f:
        digest_md = f.read()

    logging.info('Loading OpenAI API key')
    with open('secrets/openai_api_key.txt', 'r') as f:
        openai.api_key = f.read().strip()

    logging.info('Generating socials...')
    messages = [
        {'role': 'system', 'content': SOCIALS_SYSTEM_PROMPT},
        {'role': 'user', 'content': digest_md}
    ]

    socials = openai.ChatCompletion.create(
        model='gpt-4', 
        messages=messages,
        max_tokens=500,
        temperature=0
    ).choices[0]['message']['content']

    print(socials)
