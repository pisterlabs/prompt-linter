import logging
import sys
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import openai

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    download_loader
)


WikipediaReader = download_loader("WikipediaReader")
wikipedia_reader = WikipediaReader()

documents = wikipedia_reader.load_data(pages=["Cristiano_Ronaldo"])

content = documents[0].get_content().split("\n")


dataset = []

system_prompt = "You are an question maker, your task is make question from content \
You response must in format: \
<Question> --- <Answer>\n\
<Question2> --- <Answer2>"

for item in tqdm(content[:10]):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': item}
        ]
    )

    responses = response['choices'][0]['message']['content'].split('\n')

    for i in responses:
        try:
            question, answer = i.split(' --- ')

            dataset.append({
                'context': item,
                'question': question,
                'answer': answer
            })
        except Exception:
            break

dataset = pd.DataFrame.from_records(dataset)

dataset.to_csv("eval_dataset.csv", index=False)