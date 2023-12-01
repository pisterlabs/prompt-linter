"""
Loads the Iris sataset, transforms it, and then fits GPT-3-model on it.

Takes a "name" argument since I did a few variations of this script,
each saved as its own separate experiment.
"""



from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import json
import openai
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str)
args = parser.parse_args()

iris = datasets.load_iris()

transformed = 14*iris['data'] + 6
transformed = np.vectorize(round)(transformed)
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(transformed, y,
                                    test_size=0.5, stratify=y, random_state=97)


input_template = ''
for x, y in zip(x_train, y_train):
    x = ', '.join(map(str, x))
    y = str(y)
    input_template += f'Input = {x}, output = {y}\n'

results = dict()
results['input_template'] = input_template

results['x_train'] = [x.tolist() for x in list(x_train)]
results['y_train'] = [int(x) for x in list(y_train)]

results['x_test'] = [x.tolist() for x in list(x_test)]
results['y_test'] = [int(x) for x in list(y_test)]


engines = ['ada', 'babbage', 'curie', 'davinci']

for engine in engines:
    results[engine] = dict()
    results[engine]['gpt_output_raw'] = []
    results[engine]['gpt_classification'] = []


openai.api_key = os.getenv("OPENAI_API_KEY")
for engine in engines:
    for x in x_test:
        input = ', '.join(map(str, x))
        input_text = (input_template +
                        f'Input = {input}, output ='
                        )
        response = openai.Completion.create(engine=engine,
                                            prompt=input_text,
                                            max_tokens=4,
                                            temperature=0, top_p=0)
        response_text = response['choices'][0]['text']
        results[engine]['gpt_output_raw'].append(response_text)
        results[engine]['gpt_classification'].append(
            int(
                re.findall('-?\d+',response_text
                            )[0]
                )
            )

with open(f'iris_results/{args.name}.json', 'w+') as file:
    json.dump(results, file)
