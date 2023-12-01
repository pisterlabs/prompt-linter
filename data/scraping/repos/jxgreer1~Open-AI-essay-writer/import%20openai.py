import openai
openai.api_key="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"

# Define the prompt
prompt = "write a Cuban Missile crisis essay that is 5 paragraphs"

# Use the GPT-3 model to generate the essay
completions = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    temperature=0.5,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Print the generated essay
print(f = open("dist/myfile.txt", "x"))