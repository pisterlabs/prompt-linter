import os
import sys
import argparse
import openai
from github import Github

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--pr', help='review a PR', required=False)
parser.add_argument('--prompt', help='prompt for OpenAI', nargs='+', required=False)
args = parser.parse_args()


max_tokens = int(os.getenv('OPENAI_MAX_TOKENS'))

if args.pr is not None:
  g = Github(os.getenv('AAI_GITHUB_ACCESS_TOKEN'))
  repo = '/'.join(args.pr.split('/')[3:5])  # TODO: this is dumb, make it smart
  pull = int(args.pr.split('/')[-1])  # TODO: dumb-ish
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

openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Completion.create(
  model=os.getenv('OPENAI_MODEL'),
  prompt=args.prompt,
  max_tokens=max_tokens,
  temperature=float(os.getenv('OPENAI_TEMPERATURE')),
  top_p=float(os.getenv('OPENAI_TOP_P')),
  frequency_penalty=float(os.getenv('OPENAI_FREQUENCY_PENALTY')),
  presence_penalty=float(os.getenv('OPENAI_PRESENCE_PENALTY'))
)
print(response['choices'][0]['text'])
