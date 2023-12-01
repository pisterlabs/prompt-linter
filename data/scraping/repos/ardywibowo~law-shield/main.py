# import cohere  

# api_key = "kJMD23a0eXJNqY6TmBcqGjWwvI795It75kT5xfTq"

# co = cohere.Client(api_key)

# response = co.generate(
#   prompt='Please explain to me how LLMs work',
# )
# print(response)


import pandas as pd
import openai

# Initialize the API with your key
openai.api_key = "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"

prompt_path = """MAUD/tasks/MAUD_"Ability to consummate" concept is subject to MAE carveouts/base_prompt.txt"""
prompt = open(prompt_path, 'r').read()

contract_path = """MAUD/tasks/MAUD_"Ability to consummate" concept is subject to MAE carveouts/train.tsv"""
contracts = pd.read_csv(contract_path, sep='\t')

prompt.format(text=contracts.iloc[0]['text'])

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
print(completion.choices[0].message.content)
