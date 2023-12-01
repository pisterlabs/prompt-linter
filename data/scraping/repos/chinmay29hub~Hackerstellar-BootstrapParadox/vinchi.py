import openai
import json

openai.api_key = "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"
total = 3500

instructions = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"I have {total} dollars ,help me create a budget for this month for my  education, medical, investement, groceries, misc and bills for a month",
                max_tokens=200,
)


json_string = json.dumps(instructions)
json_object = json.loads(json_string)

generated_text = json_object['choices'][0]['text'].strip()

print(generated_text)
#
# print(instructions)