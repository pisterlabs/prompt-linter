from dotenv import load_dotenv
import os
import cohere

# configuration
load_dotenv()
secret_key = os.getenv("API_KEY")
co = cohere.Client(secret_key)

# Topics the user can choose from
# examples = ["In less than 60 words, how can I relax in the office?",
#             "In less than 60 words, what physical activities can I do in the office?"
#             "In less than 60 words, how can I calm myself down in the office?"]

# FEEDING MY AI PET THE TOPIC SO THEY CAN ANSWER IT HAHAHAA
def genActivity(topic):
    response = co.generate(
    model='command',
    prompt=topic,
    max_tokens=300,
    temperature=0.3,
    )
    # response is a string that is the answer to the topic in less than 60 words
    return response.generations[0].text

# print(genActivity(topic))