import argparse
import openai
import json

# Define command line arguments
parser = argparse.ArgumentParser(description='Tool for communicating with OpenAI API')
parser.add_argument('-p', '--path', help='Path to messages file', required=True)
parser.add_argument('-k', '--key', type=str, help='API KEY', required=True)
parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='OpenAI model')

args = parser.parse_args()

with open(args.path, 'r') as f:
    messages = json.load(f)

openai.api_key = args.key

chat = openai.ChatCompletion.create(model=args.model, messages=messages)
reply: object = chat.choices[0].message.content
print(reply)
