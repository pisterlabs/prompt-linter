import openai

# Set your OpenAI API key
openai.api_key = "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"

# Provide a prompt and get a response
prompt = "find x of 2x + 2 = 10"

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=5,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n", "x="],
)

print(response)
