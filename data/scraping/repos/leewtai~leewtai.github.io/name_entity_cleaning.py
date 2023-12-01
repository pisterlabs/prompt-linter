import os
import json
from glob import glob

import openai
import pandas as pd
import requests


openai.api_key = os.getenv('OPENAI_API_KEY')
fns = glob('../h1b/data/*.csv')

int_cols = ['Initial Approval', 'Initial Denial', 'Continuing Approval',
            'Continuing Denial']


def process_h1b(h1b):
    fix = h1b.copy()
    for int_col in int_cols:
        fix[int_col] = h1b[int_col].astype(str)
        fix[int_col] = fix[int_col].str.replace(',', '')
        fix[int_col] = fix[int_col].astype(int)
    fix = fix.loc[fix.Employer.notna()]
    return fix


dfs = []
for fn in fns:
    df = pd.read_csv(fn)
    if int_cols[0] not in df.columns:
        df.rename(columns={col + 's': col for col in int_cols}, inplace=True)
    dfs.append(process_h1b(df))

df = pd.concat(dfs, ignore_index=True)

df.columns
df['Fiscal Year'].value_counts()
df.head(3)
employers = df.Employer[df['Fiscal Year'] == 2022].unique()
employers = df.Employer[df.Employer.str.contains(r"(AMAZON|GOOGLE)")].unique()
len(employers)
employers[:10]

companies = []

em_sep = ', '.join(employers[:100])
message = ("For each company listed below, if it is a subsidiary of another "
           "company, please give me the name of its parent company, otherwise "
           "don't return anything. Here is the list of companies:") + em_sep
len(message)


results = openai.ChatCompletion.create(
    model="gpt-4-0314",
    messages=[
        {"role": "user",
         "content":  message},
    ],
    temperature=0,
)

assert results.choices[0]['finish_reason'] == 'stop'

results.model
results.keys()
results.usage
# old prompt: completion: 1164, prompt: 892
# new prmopt: completion: 121, prompt: 889
results.usage.completion_tokens + results.usage.prompt_tokens

results.choices[0].message.role
#companies = results.choices[0].message.content.split('\n')

created = results.created
with open(f"companies_subsidiary_{created}.txt", "w") as f:
    f.write(results.choices[0].message.content)


results = openai.Completion.create(
    model="gpt-4",
    prompt=message,
)

