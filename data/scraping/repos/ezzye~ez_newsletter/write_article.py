import openai

openai.api_key = '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'

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Translate the following English text to French: '{}'",
    max_tokens=60
)

print(response.choices[0].text.strip())


with open("output.md", "w") as file:
    file.write("# Title\n")
    file.write(response.choices[0].text.strip() + "\n")
