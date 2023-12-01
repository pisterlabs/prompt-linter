import os
import openai
import dotenv

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

prompt = '''
Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: "Predator is the Best Movie ever made"
Sentiment:
'''

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0
)

print(response['choices'][0]['text'])
