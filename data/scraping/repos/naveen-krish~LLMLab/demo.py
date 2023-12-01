import openai


secret_key='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'
prompt='Say Hello Naveen'

openai.api_key = secret_key

output = openai.Completion.create(
    model='text-davinci-003',
    prompt=prompt,
    max_tokens=20,
    temperature=0
)
output_text = output['choices'][0]['text']
print(output_text)