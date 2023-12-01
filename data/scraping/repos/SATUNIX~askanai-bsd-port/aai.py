import os
import sys
import argparse
import openai
from github import Github
import configparser

import warnings
warnings.filterwarnings("ignore")

#LOAD CONFIG FROM CONFIG FILE
openai_model = config.get('openai', 'model')

parser = argparse.ArgumentParser()
parser.add_argument('--pr', help='review a PR', required=False)
parser.add_argument('--prompt', help='prompt for OpenAI', nargs='+', required=False)
args = parser.parse_args()

max_tokens = int(config.get('openai', 'max_tokens'))

if args.pr is not None:
  g = Github(config.get('github', 'access_token'))
  repo = '/'.join(args.pr.split('/')[3:5])
  pull = int(args.pr.split('/')[-1])
  pr = g.get_repo(repo).get_pull(pull)
  prompt_pr = ''
  for file in pr.get_files():
    prompt_pr = prompt_pr + '\n ' + file.patch
  args.prompt = 'Review the following git patch changes for any possible bugs, errors, or improvements, and provide a summary of the findings: ' + prompt_pr
  max_tokens = max_tokens - len(args.prompt)
  if max_tokens < 0:
    print('prompt was too big ('+str(len(args.prompt))+'): '+args.prompt)
    sys.exit(1)

if args.prompt is not None:
  args.prompt = ' '.join(args.prompt)

openai.api_key = config.get('openai', 'api_key')
response = openai.Completion.create(
  model=openai_model,
  prompt=args.prompt,
  max_tokens=max_tokens,
  temperature=float(config.get('openai', 'temperature')),
  top_p=float(config.get('openai', 'top_p')),
  frequency_penalty=float(config.get('openai', 'frequency_penalty')),
  presence_penalty=float(config.get('openai', 'presence_penalty'))
)
print(response['choices'][0]['text'])
