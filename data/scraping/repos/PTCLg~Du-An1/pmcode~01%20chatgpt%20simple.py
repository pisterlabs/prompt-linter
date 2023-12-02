import openai

openai.api_key = "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"

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Give me 3 ideas for apps I could build with openai apis "}])
print(completion.choices[0].message.content)