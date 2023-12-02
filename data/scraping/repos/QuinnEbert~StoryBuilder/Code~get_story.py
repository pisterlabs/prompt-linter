import time
import openai
import os

openai.api_key = os.environ['OPENAI_KEY']
prompt = os.environ['STORY_PROMPT']

while True:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        print(response.choices[0].message.content)
        break  # Exit loop if request is successful
    except Exception as e:
        time.sleep(0.1)  # Wait for 0.1 second if rate limit is hit