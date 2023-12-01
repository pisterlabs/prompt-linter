import openai
import re

# Set up the API key
openai.api_key = "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"

# Define the prompt to generate text completion
prompt = "Energy Crisis"

# Generate text completion for the prompt
completions = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Extract the generated text
generated_text = completions.choices[0].text

# Split the generated text into words
words = re.findall(r'\w+', generated_text)

# Create a set to store the unique keywords
keywords = set()

# Add each word to the set if it's not a stop word
stop_words = {"the", "and", "of", "to", "in", "that", "is", "with", "for", "on"}
for word in words:
    if word.lower() not in stop_words:
        keywords.add(word.lower())

# Convert the set of keywords to a list
keywords = list(keywords)

# Print the generated keywords
print("Generated keywords:")
for keyword in keywords:
    print("- " + keyword)

