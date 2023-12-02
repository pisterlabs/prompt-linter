import openai

openai.api_base = "http://localhost:4891/v1"
#openai.api_base = "https://api.openai.com/v1"

openai.api_key = "not needed for a local LLM"

# Set up the prompt and other parameters for the API request
prompt = "Hi"
print(prompt)

# Models
# model = "gpt-3.5-turbo"
#model = "mpt-7b-chat"
#model = "gpt4all-j-v1.3-groovy"
model = "vicuna-7b-1.1-q4_2"

# Make the API request
response = openai.Completion.create(
    model=model,
    prompt=prompt,
    max_tokens=400,
    temperature=0.7,
    top_p=0.95,
    n=1,
    echo=True,
    stream=False
)

# Print the generated completion
print(response)
