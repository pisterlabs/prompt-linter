import openai, sys

openai.api_key = sys.argv[1]

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a poem about programming",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

poem = response['choices'][0]['text'].replace('\n', '  \n')

print(poem)

header = '''
# openai-random-poem
 A poem generated daily by OpenAI

[![Poem](https://github.com/fbiego/openai-random-poem/actions/workflows/main.yml/badge.svg)](https://github.com/fbiego/openai-random-poem/actions/workflows/main.yml)

### today's poem'''

file = open('README.md', 'w')
file.write(header + poem)
file.close()
