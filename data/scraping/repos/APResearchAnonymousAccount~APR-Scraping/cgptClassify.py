import openai_secret_manager
import requests

# Get API key
secrets = openai_secret_manager.get_secret("openai")
api_key = secrets["api_key"]

# Define base URL for API request
base_url = "https://api.openai.com/v1/engines/davinci-codex/completions"

# Example tweets to classify
tweets = [
    "The economy is doing great under Trump's leadership.",
    "Climate change is real and we need to take action now.",
    "I support universal healthcare for all citizens.",
    "We need to build a wall to protect our borders.",
]

# Classify each tweet
for tweet in tweets:
    # Define data for API request
    data = {
        "prompt": f"What is the political affiliation of someone who says: {tweet}",
        "temperature": 0.5,
        "max_tokens": 100,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    # Send API request
    response = requests.post(base_url, json=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    # Print classification
    print(f"\"{tweet}\" is classified as: {response.json()['choices'][0]['text']}")
