import os
import openai
from prd_templates import prdFormat

api_key = "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"
openai.api_key = api_key

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

prompt = f"""
For the html delimited by triple backticks, write several e2e test cases for it.
The test cases must fit the format below:

Feature: Account Holder withdraws cash

Scenario:
    Given
      And
      And 
     When
     Then
      And
      And

```{prdFormat}```
"""

response = get_completion(prompt)
print("job done")
text_file = open("sample.html", "w")
write_to_file = text_file.write(response)
text_file.close()
