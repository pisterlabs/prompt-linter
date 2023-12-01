import os
import openai
import json
import keys as k
openai.organization = "org-gsiX34pWAdwNfeWAkY6Rejg0"
openai.api_key = k.openai_key # os.getenv("OPENAI_API_KEY")
# engines = openai.Engine.list()
# for e in engines.data:
#     print(e.id)


# completion = openai.Completion.create(engine="ada", prompt="italian recipe")
#
# print("the completion:")
# print(completion.choices[0].text)


# response = openai.Completion.create(
#   model="code-davinci-002",
#   prompt="write the first 10 numbers in Java",
#   temperature=0,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )
# print()
# print(response.choices[0].text)

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="alle österreichischen Bundesländer als JSON", #
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response.choices[0].text)
j_str = response.choices[0].text
j = json.loads(j_str)

for b in j:
    print("%s: %s" % (b["name"], b["capital"]))



