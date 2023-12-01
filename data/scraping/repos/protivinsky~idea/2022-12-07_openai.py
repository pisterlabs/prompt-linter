import openai

openai.api_key = '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'

engines = openai.Engine.list()

completion = openai.Completion.create(engine='davinci', prompt='Write a program to generate prime numbers in Python.')
print(completion.choices[0].text)

157.5 + 1007.5 + 75 + 171

