import openai

# Function to generate a slogan for an e-commerce website that sells eco-friendly products
# using OpenAI's GPT-3 API

def generate_slogan(api_key):
    # Set the API key
    openai.api_key = api_key

    # Set the prompt
    prompt = "Generate a catchy slogan for an e-commerce website that sells eco-friendly products"

    # Generate slogan suggestions
    slogan_suggestions = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=100,
        n=5,
        temperature=0.7
    )

    # Select the best slogan
    best_slogan = slogan_suggestions.choices[0].text.strip()

    return best_slogan