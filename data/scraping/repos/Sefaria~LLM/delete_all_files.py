import openai
from openai.error import TryAgain
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    files = openai.File.list()
    for file in files['data']:
        print('Deleting', file.id)
        try:
            openai.File.delete(file.id)
            openai.File.download()
        except TryAgain:
            print("skip")
