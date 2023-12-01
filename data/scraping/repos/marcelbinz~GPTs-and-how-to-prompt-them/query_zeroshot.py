import openai
import numpy as np
import pandas as pd
import time

def query(text, engine="gpt-3.5-turbo"):
    openai.api_key = "YOUR KEY"

    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model = engine,
                messages = [{"role": "user", "content": text}],
                max_tokens = 100,
                temperature = 0.0,
                stop = ["\n", ' '],
            )
            return response.choices[0].message.content
        except:
            print(f'error, {i}th loop')
            time.sleep(1)
            pass

problems = pd.read_csv('data/problems.csv')

data = []
for r in range(problems['run'].max() + 1):
    for e in range(problems['example'].max() + 1):
        problem = problems[(problems['run'] == r) & (problems['example'] == e)]
        prompt = '+'.join(str(problem[e].item()) for e in problem.filter(regex='digit_')) + '='
        print(prompt)

        response = query(prompt)
        print(response)

        row = [problem['run'].item(),
            problem['example'].item(),
            problem['digit_0'].item(),
            problem['digit_1'].item(),
            problem['digit_2'].item(),
            problem['digit_3'].item(),
            problem['digit_4'].item(),
            problem['digit_5'].item(),
            problem['digit_6'].item(),
            problem['digit_7'].item(),
            problem['digit_8'].item(),
            problem['digit_9'].item(),
            problem['result'].item(),
            response
        ]
        data.append(row)

df = pd.DataFrame(data, columns=['run', 'example', 'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9', 'result', 'response'])
df.to_csv('results/zeroshot_experiment.csv')
