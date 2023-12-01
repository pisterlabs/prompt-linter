import openai
import random

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Define a prompt for GPT-3 to generate a headline
prompt = "Generate a headline for a news article about a major world event"

# Set the model ID for GPT-3
model = "text-davinci-002"

# Generate a headline using GPT-3
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=50,
    n=1,
    stop=None,
    temperature=0.5,
)

# Get the generated headline from the response
headline = response.choices[0].text.strip()

# Print the headline to the console
print("Generated headline:", headline)
