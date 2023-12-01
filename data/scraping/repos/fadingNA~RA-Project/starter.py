import openai

api_key = "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"
openai.api_key = api_key
new_prompt = "Who is CAA in Canada"
answer = openai.Completion.create(
    model="text-davinci-003",
    prompt=new_prompt
)

print(answer['choices'][0]['text'])

new_prompt = """ Which type of gas is utilized by plants during the process of photosynthesis?"""
answer = openai.Completion.create(
    model="text-davinci-003",
    prompt=new_prompt
)

print(answer['choices'][0]['text'])
