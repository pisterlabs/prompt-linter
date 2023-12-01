import openai

openai.api_key = "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"

a = input("tell me about someone you like: ")
b = " ->"
input = a + b

response = openai.Completion.create(
    model="davinci:ft-personal-2023-01-15-05-04-22",
    prompt=input,
    max_tokens=30,
    temperature=0.5,
    n=10,
    best_of=12,


)
# only prints the required text from json file
output = (response['choices'][0]['text'])

for element in output:
    if element == ".":
        break
    print(element, end="")
