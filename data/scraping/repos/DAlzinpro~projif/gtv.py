import openai

import os
os.environ["OPENAI_API_KEY"] = "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"
print("gay")
perg = input("")
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": perg}])
print(completion.choices[0].message.content)
