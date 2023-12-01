import os
import openai

openai.organization = "org-9PKnn7txwxhPDTNvTe3ZL164"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Example1 : Get a list of available models
# List available models
models = openai.Model.list()
print("Available models:")
for model in models["data"]:
    print(model["id"])


# Example2 : Get a list of available engines
# Get joke from OpenAI API
prompt = "Tell me a joke"
response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=100
)

# Print joke
print("\nJoke:\n" + "-" * 20)
print(response.choices[0].text.strip())
