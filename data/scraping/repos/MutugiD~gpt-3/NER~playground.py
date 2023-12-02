import os
import openai

openai.api_key = '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'

date = "29th January 2020" #01-05-2015"
price = "$450000"
use = "Co-working office"
area = "1,000 square meters"

outprompt = 'Date: %s\nPrice: %s\nUse: %s\nArea: %s\n\nOutput: ' % (date, price, use, area)

completion = openai.Completion.create(
                        model="davinci:ft-personal:ner-generator-2023-01-21-10-07-15", 
                        prompt= outprompt,
                        top_p=1,
                        max_tokens=500,
                        stop=["\n###", "\n\n"])

print(completion["choices"][0]["text"])
