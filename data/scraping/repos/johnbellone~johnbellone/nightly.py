from dotenv import dotenv_values
from linkedin_api import Linkedin
import openai

config = dotenv_values(".env")

# Authenticate using any Linkedin account credentials
linkedin = Linkedin(config['LINKEDIN_USER'], config['LINKEDIN_PASS'])
openai.api_key = config['OPENAI_API_KEY']

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write something interesting about a super hero animal that is a dog."},
    ]
)

print(response)
print(linkedin.get_user_profile())
