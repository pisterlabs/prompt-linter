from openai import AzureOpenAI
import re
import numpy as np
import pandas as pd

def clean_article_text(input_file, output_file):
    with open("SECRETKEY.txt", "r") as f:
        key = f.read()
        key = key.split('\n')[0]

    client = AzureOpenAI(api_key=key,
    azure_endpoint='https://api.umgpt.umich.edu/azure-openai-api/ptu',
    api_version='2023-03-15-preview')

    indf = pd.read_csv(input_file)
    indf = indf.replace("", np.NaN)
    indf = indf.dropna()

    outdict = {"index": [], "text": [], "url": []}

    for index, row in indf.iterrows():
        text = re.sub(r'[^a-zA-Z0-9\._-]', '', row['text'])

        print(f"this raw article contains {len(text)} characters")
        final_response = np.NaN
        if (len(text) < 1000):
            print(text) #print to understand why there is an error
        else:

            text = text[:min(len(text), 31000)]

            response = client.chat.completions.create(model='gpt-4',
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Remove words that don't relate to the main themes of to the article deliminated by triple quotes\n" + f'"""{text}"""'}
            ])
            final_response = response.choices[0].message.content

        outdict["index"].append(index)
        outdict["text"].append(final_response)
        outdict["url"].append(row['url'])

        print(f"article {index} done")
    # print the response
    outdf = pd.DataFrame(data = outdict)
    outdf.to_csv(output_file)