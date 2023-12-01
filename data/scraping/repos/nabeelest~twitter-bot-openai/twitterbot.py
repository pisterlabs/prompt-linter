import tweepy
import openai
import random
import os

AI_KEY = os.environ["ai_key"]
api_key = os.environ["api_key"]
api_secret = os.environ["api_secret"]
access_token = os.environ["access_token"]
access_token_secret = os.environ["access_token_secret"]
bearer_token = os.environ["bearer_token"]

client = tweepy.Client(
    consumer_key=api_key,
    consumer_secret=api_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

auth = tweepy.OAuth1UserHandler(api_key, api_secret,access_token, access_token_secret)
api = tweepy.API(auth)

openai.api_key = AI_KEY

words = ["how bad i miss her","being hopeful about her","Melodramatic feeling about her","heartbreak","romantic feelings","Enigmatic feeling","Heartfelt feelings about her","Mystical feelings","Contemplative thought","being Cynical","Thought-provoking thing","Whimsical thoughts","Somberness","Nostalgia","Introspecting oneself","Wistful thought","Transcendent thought","Poignant thought","feeling Vulnerable","Melancholia","Connections with people","Resilience","carl jung's philosophy","solitude","social media detox","mahmoud darwish's philosophy","albert camus's philosophy","Alan Watts philosophy","philosophy of love","philosophy of heartbreak","life in general","the good old days","poetry","teenage"]
category = random.choice(words)

# Define your prompt
prompt = f"tweet a quote from a movie with movie name. don't add any hashtags, quotations or special characters and try writing in no-caps."


# Call the API
response = openai.Completion.create(
  engine='text-davinci-003',
  prompt=prompt,
  max_tokens=100,
  n=1,
  stop=None,
  temperature=1.0
)

# Extract the generated text from the API response
generated_text = response.choices[0].text.strip()

print(generated_text)

tweet = client.create_tweet(
            text=generated_text
            )
