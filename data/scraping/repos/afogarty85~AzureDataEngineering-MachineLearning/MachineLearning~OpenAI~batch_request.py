import pandas as pd
from tenacity import (retry, stop_after_attempt,  wait_random_exponential)
import openai
import numpy as np

# set requirements
prompt_prefix = 'Summarize:\n'
prompt_ending = '\n\n###\n\n'

# load data
df = pd.read_parquet(r'data/df.parquet')

# add in prompt
df['Prompt'] = prompt_prefix + df['Title'] + prompt_ending

# openai keys
openai.api_type = "azure"
openai.api_base = "https://endpoint.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = ''

# find chunk size to ensure batches of 20
chunk_size = min([i for i in range(1, 10000) if ((df.shape[0] // i) < 20) ])

# split prompts into batches
prompt_list = np.array_split(df['Prompt'].tolist(), chunk_size, axis=0)
print(f"Using this many batches: {len(prompt_list)}")

# split primary key into batches
key_list = np.array_split(df['IncidentId'].tolist(), chunk_size, axis=0)

assert len(prompt_list[0]) <= 20, 'error batching'
assert len(prompt_list[-1]) <= 20, 'error batching'

# create backoff retry
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

# holding
storage_df = pd.DataFrame()

# loop over batches
for i, (batch, key) in enumerate(zip(prompt_list, key_list)):
    print(f"Now on iter: {i}")

    try:

        # send request
        resp = completion_with_backoff(
            engine="davinci_mine",
            prompt=batch.tolist(),
            max_tokens=128,
            temperature=1,  # as temp decreases, more likely tokens increase
            top_p=1, # if 1, always choose most likely token
            best_of=1,
        )

        # match completions to prompts by index
        batch_response = [""] * len(batch)
        batch_key = []
        for choice in resp.choices:
            # store resp
            batch_response[choice.index] = choice.text
            # store primary key
            batch_key.append(key[choice.index])
        
        # todf
        tdf = pd.DataFrame({
                            'response': batch_response,  
                            'key': batch_key,
                            })

        # store results
        storage_df = pd.concat([storage_df, tdf], axis=0)

    except Exception as e:
        print("Exception found on iter {i}, {e}")


storage_df.to_parquet('./data/augmented_results.parquet')

