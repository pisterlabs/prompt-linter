import os
import openai

import pprint

pp = pprint.PrettyPrinter(indent=1)

openai.api_key = os.getenv("OPENAI_API_KEY")

print(f'Please type in your product review')
review = input(">")

prompt = "Read this customer response then answer the following questions:\n\n"

#print(f'Now please ask me a question about the review')
#question = input("#")
question = "What was the product? Was the customer satisfied? What was the problem? Was the reviwer polite? What are the asking to be done?"
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=f'{prompt}{review}\n\nQuestions:{question}\n\nAnswers:1',
  temperature=0.1,
  max_tokens=128,
  top_p=1,
  frequency_penalty=0.37,
  presence_penalty=0,
  stop=["\n\n"]
)
#pp.pprint(response)
finish = response.choices[0].text
print(f'{finish}')
