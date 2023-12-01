import openai

# Set up the OpenAI API key
openai.api_key = "<YOUR API HERE>"

# Ask the user for the prompt
prompt = input("Enter a prompt: ")

# Use the `Completion` endpoint to generate text based on the prompt
completion = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Extract the generated text
generated_text = completion.choices[0].text

# Print the generated text
print("\nGenerated text:")
print(generated_text)



