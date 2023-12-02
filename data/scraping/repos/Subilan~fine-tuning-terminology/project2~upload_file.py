from apikey import OPENAI_API_KEY
import openai
import os
import argparse
openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser()

parser.add_argument('filenames', nargs='*')

args = parser.parse_args()

filenames = args.filenames

def upload(filename):
    result = openai.File.create(
        file=open(f'{filename:s}', 'rb'),
        purpose='fine-tune'
    )
    return result.id
    
for name in filenames:
    print(f'upload: {name:s} -> {upload(name):s}')
